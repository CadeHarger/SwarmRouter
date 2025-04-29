import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import ioh
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count

import sys
import argparse
import warnings
import os

import pandas as pd

import numpy as np

from copy import copy

from niapy.task import Task
from niapy.problems import Problem
import niapy.algorithms as nialg
from niapy.algorithms.basic import *

from mealpy_helper import get_models

DATA_FOLDER = "test"
MAX_THREADS = 32

def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(MAX_THREADS, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results



class NIA_Problem(Problem):
    def __init__(self, ioh_problem, *args, **kwargs):
        super().__init__(ioh_problem.meta_data.n_variables, ioh_problem.bounds.lb[0], ioh_problem.bounds.ub[0], *args, **kwargs)
        self.f_internal = ioh_problem

    def _evaluate(self, x):
        return self.f_internal(x)

    def _reset(self):
        self.f_internal.reset()


class Algorithm_Evaluator_niapy():
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps):
        my_problem = NIA_Problem(func)
        for seed in range(n_reps):
            np.random.seed(int(seed))
            task = Task(problem=my_problem, max_evals=10000*func.meta_data.n_variables)
            algo = eval(f"{self.alg}()")
            algo.run(task)
            func.reset()
            my_problem._reset()          
        
class Algorithm_Evaluator_mealpy():
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps):
        term = {
          "max_fe": func.meta_data.n_variables * 10000
        }
        
        problem = {
            "fit_func": func,
            "lb": func.bounds.lb,
            "ub": func.bounds.ub,
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }
        for seed in range(n_reps):
            np.random.seed(int(seed))
            self.alg.solve(problem, termination=term)
            func.reset()

def run_optimizer_niapy(temp):
    
    algname, fid, iid, dim = temp
    print(algname, fid, iid, dim)

    algorithm = Algorithm_Evaluator_niapy(algname)

    logger = ioh.logger.Analyzer(root=f"{DATA_FOLDER}/NIAPY/", folder_name=f"{algname}_F{fid}_I{iid}_{dim}D", algorithm_name=f"{algname}")

    func = ioh.get_problem(fid, dimension=dim, instance=iid, problem_class=ioh.ProblemClass.REAL)
    # func.enforce_bounds(np.inf)
    func.attach_logger(logger)
    
    algorithm(func, 5)
    
    logger.close()
    
def run_optimizer_mealpy(temp):
    
    alg, fid, iid, dim = temp
    print(alg.name, fid, iid, dim)
    model = copy(alg)
    algorithm = Algorithm_Evaluator_mealpy(model)

    logger = ioh.logger.Analyzer(root=f"{DATA_FOLDER}/mealpy/", folder_name=f"{model.name}_F{fid}_I{iid}_{dim}D", algorithm_name=f"{model.name}")

    func = ioh.get_problem(fid, dimension=dim, instance=iid, problem_class=ioh.ProblemClass.REAL)
    # func.enforce_bounds(np.inf)
    func.attach_logger(logger)
    
    algorithm(func, 5)
    
    logger.close()    

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    fids = range(1,25)
    
    algnames = nialg.basic.__all__

    iids = range(1,11)
    
    dims = [2,5,10,20]
    
    models = get_models()
    args = product(models, fids, iids, dims)
    runParallelFunction(run_optimizer_mealpy, args)
    
    args = product(algnames, fids, iids, dims)

    runParallelFunction(run_optimizer_niapy, args)


