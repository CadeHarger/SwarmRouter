import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    solution_embeds = torch.stack(torch.load("solution_embeds.pt", map_location=torch.device('cpu'))).squeeze().float()
    problem_embeds = torch.stack(torch.load("problem_embeds.pt", map_location=torch.device('cpu'))).squeeze().float()

    print(solution_embeds.shape)
    print(problem_embeds.shape)


    # Load the 'labels' as a shape (problems, dim, embed_dim)
    performance_data = pd.read_csv("./Processed_data/AUC_mealpy.csv")

    # Average over instances and runs
    performance_data = performance_data.groupby(['algname', 'fid', 'dim'])['auc'].mean().reset_index()

    # Get the order of the algorithms in the embedding vector
    with open("solution_algorithms.txt", "r") as file:
        algorithms = '\n'.join(file.readlines())
    algorithms = [alg for alg in algorithms.split('<END_OF_ALGORITHM>') if len(alg.strip()) > 1]
    algnames = [alg.split('class ')[1].split('(')[0] for alg in algorithms]
    print(len(algnames))

    dim_options = [2, 5, 10, 20]
    normal_factors = np.zeros((problem_embeds.shape[0], len(dim_options)))
    unnormalized_labels = np.zeros((problem_embeds.shape[0], len(dim_options), problem_embeds.shape[-1]))
    for i, row in performance_data.iterrows():
        try:
            sorted_index = algnames.index(row['algname'])
        except ValueError:
            continue
        scaled_vec = solution_embeds[sorted_index].numpy() * row['auc'] # Scale the vector by the performance
        normal_factors[row['fid']-1, dim_options.index(row['dim'])] += row['auc']
        unnormalized_labels[row['fid']-1, dim_options.index(row['dim']), :] += scaled_vec

    # Original labels with shape (problems, dim, embed_dim)
    original_labels = unnormalized_labels / normal_factors[:, :, None]
    print("Original labels shape:", original_labels.shape)


    # Create one-hot vectors for dimensions
    num_dims = len(dim_options)
    dim_one_hot = torch.eye(num_dims).float()  # Shape: (num_dims, num_dims)

    # Expand problem embeddings to include all dimension combinations
    # Shape: (problems * num_dims, problem_embed_dim + num_dims)
    expanded_problem_embeds = []
    for i in range(problem_embeds.shape[0]):
        for d in range(num_dims):
            # Concatenate problem embedding with dimension one-hot vector
            combined = torch.cat([problem_embeds[i], dim_one_hot[d]])
            expanded_problem_embeds.append(combined)
    expanded_problem_embeds = torch.stack(expanded_problem_embeds)

    # Flatten labels for training
    # Shape: (problems * num_dims, embed_dim)
    flattened_labels = original_labels.reshape(-1, original_labels.shape[-1])
    print("flattened_labels.shape:", flattened_labels.shape)

    # Train the model
    test_size = 16
    
    train_labels = torch.tensor(flattened_labels[:-test_size], dtype=torch.float32)
    train_problem_embeds = expanded_problem_embeds[:-test_size]
    train_solution_embeds = solution_embeds[:-test_size]

    test_labels = torch.tensor(flattened_labels[-test_size:], dtype=torch.float32)
    test_problem_embeds = expanded_problem_embeds[-test_size:]
    test_solution_embeds = solution_embeds[-test_size:]

    model = nn.Sequential(
        nn.Linear(expanded_problem_embeds.shape[-1], 10000),
        nn.ReLU(),
        nn.Linear(10000, 10000),
        nn.ReLU(),
        nn.Linear(10000, solution_embeds.shape[-1]),
    )
    print(f"Number of parameters: {format(sum(p.numel() for p in model.parameters()), ',d')}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 1
    batch_size = 6
    train_history = []
    test_history = []
    for _ in range(epochs):
        test_loss = 0
        for i in range(0, len(train_problem_embeds), batch_size):
            print(i)
            problem_embed = train_problem_embeds[i:i+batch_size]
            solution_embed = train_solution_embeds[i:i+batch_size]
            label = train_labels[i:i+batch_size]

            # Forward pass
            pred = model(problem_embed)
            loss = criterion(pred, label)

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_history.append(loss.item())
            test_history.append(test_loss) # do it here so we match shape

        # Test the model
        with torch.no_grad():
            pred = model(test_problem_embeds)
            loss = criterion(pred, test_labels)
            test_loss = loss.item()
            test_history.append(loss.item())

    print(f"Final loss: {loss.item()}")
    

    plt.plot(train_history[3:], label="Train")
    plt.plot(test_history, label="Test")
    plt.legend()
    plt.show()


    # Evaluate the model on actual algorithm selection:

    model.eval()
    with torch.no_grad():
        pred = model(test_problem_embeds)

    print(pred.shape)

    from code_embeddings import problem_ids
    id_to_name = {v: k for k, v in problem_ids.items()}

    top_k_loss = 0
    mean_loss = 0
    best_loss = 0

    best_alg = performance_data.groupby('algname').mean().sort_values(by='auc', ascending=False).index[0]
    print(f"Best algorithm: {best_alg}")

    for i, vector_embed in enumerate(pred.numpy()):
        fid = int((problem_embeds.shape[0] - (test_size / len(dim_options)) + 1) + (i // len(dim_options)))
        dim = dim_options[i % len(dim_options)]
        
        # Calculate cosine similarity between predicted embedding and all solution embeddings
        similarities = np.dot(solution_embeds.numpy(), vector_embed) / (
            np.linalg.norm(solution_embeds.numpy(), axis=1) * np.linalg.norm(vector_embed)
        )
        
        k = 3
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        #print(f"Top {k} algorithms for problem {fid} {id_to_name[fid]} dim {dim}:")
        k_aucs = []
        for idx in top_k_indices:
            auc = performance_data[
                (performance_data['algname'] == algnames[idx]) & 
                (performance_data['dim'] == dim) &
                (performance_data['fid'] == fid)
            ]['auc']
            k_aucs.append(auc.mean())
            #print(f"- {algnames[idx]}: {similarities[idx]:.4f} (AUC: {auc.mean():.4f})")

        best_auc = performance_data[
            (performance_data['algname'] == best_alg) &
            (performance_data['dim'] == dim) &
            (performance_data['fid'] == fid)
        ]['auc'].mean()

        #print("---")
        #print(f" Actual top {k} algorithms for problem {fid} {id_to_name[fid]} dim {dim}:")
        auc = performance_data[
            (performance_data['dim'] == dim) &
            (performance_data['fid'] == fid)
        ].sort_values(by='auc', ascending=False)['auc']
        mean_loss += auc.iloc[0] - auc.mean()
        top_k_loss += auc.iloc[0] - np.mean(k_aucs)
        best_loss += auc.iloc[0] - best_auc
        #print(auc.head(k))
        #print()

    print(f"Top {k} loss: {top_k_loss / test_size}")
    print(f"Mean loss: {mean_loss / test_size}")
    print(f"Best loss: {best_loss / test_size}")

if __name__ == "__main__":
    main()
