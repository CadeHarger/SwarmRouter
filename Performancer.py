import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset


def main():
    solution_embeds = torch.stack(torch.load("solution_embeds.pt", map_location=torch.device('cpu'))).squeeze().float()
    problem_embeds = torch.stack(torch.load("problem_embeds.pt", map_location=torch.device('cpu'))).squeeze().float()

    print(solution_embeds.shape)
    print(problem_embeds.shape)


    # Load the 'labels' as a shape (problem, dim, solution)
    performance_data = pd.read_csv("./Processed_data/AUC_mealpy.csv")

    # Average over instances and runs
    performance_data = performance_data.groupby(['algname', 'fid', 'dim'])['auc'].mean().reset_index()

    # Get the order of the algorithms in the embedding vector
    with open("solution_algorithms.txt", "r") as file:
        algorithms = '\n'.join(file.readlines())
    algorithms = [alg for alg in algorithms.split('<END_OF_ALGORITHM>') if len(alg.strip()) > 1]
    algnames = [alg.split('class ')[1].split('(')[0] for alg in algorithms]
    print(len(algnames))

    # Reduce the solution embeddings to only include those with performance data
    included_algs = []
    reduced_solution_embeds = []
    for alg in algnames:
        if alg in performance_data['algname'].unique():
            included_algs.append(alg)
            reduced_solution_embeds.append(solution_embeds[algnames.index(alg)])
    print(len(included_algs))
    solution_embeds = torch.stack(reduced_solution_embeds)

    # Construct the labels
    dim_options = [2, 5, 10, 20]
    labels = np.zeros((problem_embeds.shape[0], len(dim_options), solution_embeds.shape[0]))
    for i, row in performance_data.iterrows():
        try:
            labels[row['fid']-1, dim_options.index(row['dim']), included_algs.index(row['algname'])] = row['auc']
        except ValueError:
            continue
        
    # Original labels with shape (problems, dim, embed_dim)
    original_labels = labels.copy()
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
    # Shape: (problems * num_dims, solutions)
    flattened_labels = original_labels.reshape(-1, original_labels.shape[-1])
    print("flattened_labels.shape:", flattened_labels.shape)

    # Train the model
    test_size = 1300

    embedding_pairs = torch.zeros((problem_embeds.shape[0] * num_dims * solution_embeds.shape[0], expanded_problem_embeds.shape[-1] + solution_embeds.shape[-1]))
    labels = torch.zeros((problem_embeds.shape[0] * num_dims * solution_embeds.shape[0], 1))
    infos = [] # (fid, dim, alg)
    for p in range(problem_embeds.shape[0]):
        for d in range(num_dims):
            for s in range(solution_embeds.shape[0]):
                infos.append((p + 1, dim_options[d], included_algs[s]))
                embedding_pairs[p*num_dims*solution_embeds.shape[0] + d*solution_embeds.shape[0] + s] = torch.cat((expanded_problem_embeds[p*num_dims + d], solution_embeds[s]), dim=0)
                labels[p*num_dims*solution_embeds.shape[0] + d*solution_embeds.shape[0] + s] = original_labels[p, d, s]

    print(embedding_pairs.shape)
    print(labels.shape)

    train_embedding_pairs = embedding_pairs[:-test_size]
    train_labels = labels[:-test_size]
    test_embedding_pairs = embedding_pairs[-test_size:]
    test_labels = labels[-test_size:]

    # Shuffle the training data
    shuffle_idx = np.random.permutation(len(train_embedding_pairs))
    train_embedding_pairs = train_embedding_pairs[shuffle_idx]
    train_labels = train_labels[shuffle_idx]

    model = nn.Sequential(
        nn.Linear(train_embedding_pairs.shape[-1], 1000),
        nn.ReLU(),
        nn.Linear(1000, 1)
    )
    print(f"Number of parameters: {format(sum(p.numel() for p in model.parameters()), ',d')}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000005)  # Increased learning rate

    epochs = 3
    batch_size = 128
    train_history = []
    test_history = []
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        test_loss = 0
        for i in range(0, len(train_embedding_pairs), batch_size):
            embedding_pair = train_embedding_pairs[i:i+batch_size]
            label = train_labels[i:i+batch_size]

            # Forward pass
            pred = model(embedding_pair)
            loss = criterion(pred, label)

            # Backward pass and update. Also clip the gradients to prevent exploding gradients
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if i % 1024 == 0 and i != 0:
                print(f"Epoch {epoch}, Batch {i//batch_size}")
                print(f"Predictions mean: {pred.mean().item():.4f}, min: {pred.min().item():.4f}, max: {pred.max().item():.4f}")
                print(f"Labels mean: {label.mean().item():.4f}, min: {label.min().item():.4f}, max: {label.max().item():.4f}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Progress: {format(i / len(train_embedding_pairs) * 100, '.2f')}%")
                print("---")

                # Test the model
                with torch.no_grad():
                    model.eval()
                    test_loss = 0
                    num_test_batches = max(1, len(test_embedding_pairs) // batch_size)  # Ensure at least 1 batch
                    for i in range(0, len(test_embedding_pairs), batch_size):
                        embedding_pair = test_embedding_pairs[i:i+batch_size]
                        label = test_labels[i:i+batch_size]
                        pred = model(embedding_pair)
                        pred = torch.clamp(pred, min=0, max=1)  # Clip negative values to 0
                        loss = criterion(pred, label)
                        test_loss += loss.item()
                    test_loss /= num_test_batches
                    test_history.append(test_loss)
                    model.train()

            train_history.append(loss.item())
            test_history.append(test_loss) # do it here so we match shape

        shuffle_idx = np.random.permutation(len(train_embedding_pairs))
        train_embedding_pairs = train_embedding_pairs[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

    print("Trained all")
    model.eval()

    print(f"Final loss: {loss.item()}")
    

    plt.plot(train_history[3:], label="Train")
    plt.plot(test_history, label="Test")
    plt.legend()
    plt.show()


    # Contextualize the model
    random_indices = np.random.randint(low=0, high=len(performance_data), size=(2000,))
    losses = []
    for i in random_indices:
        perf = performance_data.iloc[i]
        problem = torch.cat((problem_embeds[perf['fid']-1], dim_one_hot[dim_options.index(perf['dim'])]))
        try:
            solution = solution_embeds[included_algs.index(perf['algname'])]
        except ValueError:
            continue
        embedding_pair = torch.cat((problem, solution)).unsqueeze(0)  # Concatenate first, then add batch dimension
        pred = model(embedding_pair)
        pred = torch.clamp(pred, min=0, max=1)
        #print(f"Problem {perf['fid']}, Dimension {perf['dim']}, Algorithm {perf['algname']}, Predicted AUC: {pred.item():.4f}, Actual AUC: {perf['auc']:.4f}, Loss: {abs(pred.item() - perf['auc']):.4f}")
        losses.append(abs(pred.item() - perf['auc']))

    print(f"Average loss: {np.mean(losses):.4f}")


    predictions = np.zeros((len(embedding_pairs),))
    for i in range(0, len(embedding_pairs), batch_size):
        with torch.no_grad():
            embedding_pair = embedding_pairs[i:i+batch_size]
            label = labels[i:i+batch_size]
            pred = model(embedding_pair)
            pred = torch.clamp(pred, min=0, max=1)
            predictions[i:i+batch_size] = pred.numpy().squeeze()
    print("Predicted all")

    test_losses = []
    labels = labels.numpy()
    for i in range(0, len(embedding_pairs), len(solution_embeds)):
        max_idx = np.argmax(labels[i:i+len(solution_embeds)], axis=-1)
        top_k_prediction = predictions[i:i+len(solution_embeds)][max_idx]
        top_k_auc = labels[i:i+len(solution_embeds)][max_idx]
        test_losses.append(abs(top_k_prediction - top_k_auc))

    print(f"Average test loss: {np.mean(test_losses):.4f}")




    '''
    for i, (embedding_pair, label, info) in enumerate(zip(embedding_pairs, labels, infos)):
        if i % 1000 == 0:
            print(f"Info: {info}")
            print(f"Embedding pair: {embedding_pair}")
            print(f"Label: {label}")
            print("---")
    '''




if __name__ == "__main__":
    main()
