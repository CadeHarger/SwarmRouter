import os

# Returns a list of string SI algorithms
def get_solution_algorithms():

    path = "./venv/lib/python3.12/site-packages/mealpy"

    folders = ['bio_based', 'evolutionary_based', 'human_based', 'math_based', 'music_based', 'physics_based', 'swarm_based', 'system_based']
    algorithms = []


    for folder in folders:
        for file in os.listdir(f"{path}/{folder}"):
            if file.endswith(".py"):
                with open(f"{path}/{folder}/{file}", "r") as file:
                    lines = file.readlines()
                    start = None
                    for i, line in enumerate(lines):
                        if "(Optimizer):" in line and "class" in line:
                            if start is None:
                                start = i
                            else: 
                                algorithms.append(lines[start:i])
                    if start is not None:
                        algorithms.append(lines[start:])

    print(len(algorithms))
    return algorithms


if __name__ == "__main__":
    algorithms = get_solution_algorithms()