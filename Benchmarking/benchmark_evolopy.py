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
import evolopy.PSO
import evolopy.MVO
import evolopy.GWO
import evolopy.MFO
import evolopy.CS
import evolopy.BAT
import evolopy.WOA
import evolopy.FFA
import evolopy.SSA
import evolopy.GA
import evolopy.HHO
import evolopy.SCA
import evolopy.JAYA
import evolopy.DE 

import time

DATA_FOLDER = ""
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


class Algorithm_Evaluator():
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps):
        for seed in range(n_reps):
            np.random.seed(int(seed))
            eval(f"evolopy.{self.alg}.{self.alg}(func, -5, 5, {func.meta_data.n_variables}, 20, 1000)")
            func.reset()
        
def run_optimizer(temp):
    
    algname, fid, iid, dim = temp
    print(algname, fid, iid, dim)
    
    algorithm = Algorithm_Evaluator(algname)

    logger = ioh.logger.Analyzer(root=f"{DATA_FOLDER}/EVOLOPY/", folder_name=f"{algname}_F{fid}_I{iid}_{dim}D", algorithm_name=f"{algname}")

    func = ioh.get_problem(fid, dimension=dim, instance=iid)
    func.attach_logger(logger)
    
    algorithm(func, 5)
    
    logger.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    fids = range(1,25)
    
    # algnames = ['modde']
    algnames = ['GA',"SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"]

    iids = range(1,11)
    
    dims = [2,5,10,20]
    
    args = product(algnames, fids, iids, dims)

    runParallelFunction(run_optimizer, args)
