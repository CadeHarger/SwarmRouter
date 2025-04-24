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

import time
from copy import deepcopy
import nevergrad as ng
import nevergrad.common.typing as tp

from nevergrad.optimization.optimizerlib import (
    RCobyla,
    DiagonalCMA,
    RandomSearch, 
    PSO, 
    OnePlusOne, 
    Powell, 
    MultiBFGS,
    DE
)

from modde import ModularDE
from modcma import ModularCMAES

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

modde_params = { 'base' : {'mutation_base' : 'rand', 
                          'mutation_reference' : None,  
                          'lpsr' : False, 
                          'lambda_' : 10,
                          'use_archive' : False},
               'lshade' : {'mutation_base' : 'target', 
                          'mutation_reference' : 'pbest',  
                          'lpsr' : True, 
                          'lambda_' : 18,
                          'use_archive' : True, 
                          'adaptation_method_F' : 'shade', 
                          'adaptation_method_CR' :  'shade'}}

modcma_params = { 'base' : {},
                  'bipop' : {
                  'local_restart' : 'BIPOP'
                  }
}


class Algorithm_Evaluator():
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps):

        for seed in range(n_reps):
            np.random.seed(int(seed))
            
            if self.alg[:2] == 'ng':
                parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(-5, 5)
                optimizer = eval(f"{self.alg[3:]}")(
                    parametrization=parametrization, budget=int(10000*func.meta_data.n_variables)
                )
                optimizer.minimize(func)
                
            elif self.alg[:5] == 'modde':
                params = deepcopy(modde_params[self.alg[6:]])
                params['lambda_'] *= func.meta_data.n_variables
                c = ModularDE(func, bound_correction='saturate',
                             budget=int(10000*func.meta_data.n_variables), 
                             **params)
                c.run()
                
            else: #modcma
                print(self.alg)
                params = modcma_params[self.alg[7:]]
                c = ModularCMAES(func, d=func.meta_data.n_variables, bound_correction='saturate',
                             budget=int(10000*func.meta_data.n_variables),
                             x0=np.zeros((func.meta_data.n_variables, 1)), **params)
                c.run()
            
            func.reset()
        
def run_optimizer(temp):
    
    algname, fid, iid, dim = temp
    print(algname, fid, iid, dim)
    
    algorithm = Algorithm_Evaluator(algname)

    logger = ioh.logger.Analyzer(root=f"{DATA_FOLDER}/Baselines/", folder_name=f"{algname}_F{fid}_I{iid}_{dim}D", algorithm_name=f"{algname}")
    # logger = ioh.logger.Analyzer(root="Data_niapy/", folder_name=f"{algname}_F{fid}_I{iid}", algorithm_name=f"{algname}")

    func = ioh.get_problem(fid, dimension=dim, instance=iid)
    # func.enforce_bounds(np.inf)
    func.attach_logger(logger)
    
    algorithm(func, 5)
    
    logger.close()

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)

    fids = range(1,25)
    
    algnames = ["modcma_base", 
                "modde_base", 
                "modde_lshade", 
                "modcma_bipop", 
                "ng_RCobyla",
                "ng_DiagonalCMA",
                "ng_RandomSearch", 
                "ng_PSO", 
                "ng_OnePlusOne", 
                "ng_Powell", 
                "ng_MultiBFGS",
                "ng_DE"
               ]
    iids = range(1,11)
    
    dims = [2,5,10,20]
    
    args = product(algnames, fids, iids, dims)

    runParallelFunction(run_optimizer, args)
