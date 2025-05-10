import os

# Returns a list of string SI algorithms
def get_solution_algorithms():

    path = "./venv/lib/python3.12/site-packages/mealpy"

    folders = ['bio_based', 'evolutionary_based', 'human_based', 'math_based', 'music_based', 'physics_based', 'swarm_based', 'system_based']
    algorithms = {}


    for folder in folders:
        for file in os.listdir(f"{path}/{folder}"):
            if file.endswith(".py") and '__' not in file:
                with open(f"{path}/{folder}/{file}", "r") as file:
                    lines = file.readlines()
                    classlines = []
                    for i, line in enumerate(lines):
                        if "class " in line and "):" in line:
                            classlines.append(i)
                    if len(classlines) == 0:
                        continue
                    for i in range(len(classlines) - 1):
                        name = lines[classlines[i]].split("class ")[1].split("(")[0]
                        algorithms[name] = ''.join(lines[classlines[i]:classlines[i+1]])
                    # Add the last class
                    name = lines[classlines[-1]].split("class ")[1].split("(")[0]
                    algorithms[name] = ''.join(lines[classlines[-1]:])

    print(len(algorithms))

    # filter empty strings
    algorithms = {k: v for k, v in algorithms.items() if len(v.strip()) > 1}

    return algorithms


problem_ids = {
    "sphere": 1,
    "ellipsiod": 2,
    "rastrigin": 3,
    "bueche_rastrigin": 4,
    "linear_slope": 5,
    "attractive_sector": 6,
    "step_ellipsoid": 7,
    "rosenbrock": 8,
    "rosenbrock_rotated": 9,
    "ellipsiod_rotated": 10,
    "discus": 11,
    "bent_cigar": 12,
    "sharp_ridge": 13,
    "different_powers": 14,
    "rastrigin_rotated": 15,
    "weierstrass": 16,
    "schaffers10": 17,
    "schaffers1000": 18,
    "griewank_rosenbrock": 19,
    "schwefel": 20,
    "gallagher101": 21,
    "gallagher21": 22,
    "katsuura": 23,
    "lunacek_bi_rastrigin": 24
}

def get_problem_algorithms():
    path = "./bbob"
    algorithms = ['' for _ in range(len(problem_ids))]
    for file in os.listdir(path):
        name = file.split(".")[0]
        if name in problem_ids:
            with open(f"{path}/{file}", "r") as file:
                current_algorithm = ""
                lines = file.readlines()
                for line in lines:
                    current_algorithm += line
                algorithms[problem_ids[name] - 1] = current_algorithm
    return algorithms


def save_algorithms(algorithms, name):
    with open(f"{name}.txt", "w") as file:
        for algorithm in algorithms:
            file.write(f"{algorithm}\n <END_OF_ALGORITHM>\n")


def load_algorithms(file_path):
    algs = []
    with open(file_path, "r") as file:
        current_algorithm = ""
        for line in file:
            if "<END_OF_ALGORITHM>" in line:
                algs.append(current_algorithm)
                current_algorithm = ""
            else:
                current_algorithm += line + "\n"
    return algs


if __name__ == "__main__":
    algorithms = get_solution_algorithms()
    sorted_algorithms = dict(sorted(algorithms.items()))

    names, implementations = zip(*sorted_algorithms.items())
    save_algorithms(implementations, "solution_algorithms")
    

    problem_algorithms = get_problem_algorithms()
    save_algorithms(problem_algorithms, "problem_algorithms")

