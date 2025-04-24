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
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.spaces import SearchSpace

from opytimizer.optimizers.swarm import PSO, ABC, ABO, AF, AIWPSO, BA, BOA, BWO, CS, CSA, EHO, FA, FFOA, FSO, GOA, JS, KH, MFO, MRFO, NBJS, PIO, PSO, RPSO, SAVPSO, SBO, SCA, SFO, SOS,SSA, SSO, STOA, VPSO, WOA
from opytimizer.optimizers.misc import AOA, CEM, DOA, GS, HC, NDS
from opytimizer.optimizers.science import AIG, ASO, BH, EFO, EO, ESA, GSA, HGSO, LSA, MOA, MVO, WWO, WEO, WDO, TWO, TEO, SA
from opytimizer.optimizers.social import BSO, CI, ISA, MVPA, QSA, SSD
from opytimizer.optimizers.population import AEO, AO, COA, EPO, GCO, GWO, HHO, LOA, OSA, PPA, PVS, RFO
from opytimizer.optimizers.evolutionary import BSA, DE, EP, ES, FOA, GA, GHS, GOGHS, HS, IHS, IWO, NGHS, RRA, SGHS


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
        def helper(x):
            return func(x.reshape(-1))
        space = SearchSpace(30, func.meta_data.n_variables, func.bounds.lb, func.bounds.ub)
        optimizer = eval(f"{self.alg}()")
        function = Function(helper)

        for seed in range(n_reps):
            np.random.seed(int(seed))
            
            Opytimizer(space, optimizer, function).start(n_iterations=int((func.meta_data.n_variables * 10000) / 30))
            func.reset()
        
def run_optimizer(temp):
    
    algname, fid, iid, dim = temp
    print(algname, fid, iid, dim)
    
    algorithm = Algorithm_Evaluator(algname)

    logger = ioh.logger.Analyzer(root=f"{DATA_FOLDER}/OPYTIMIZER/", folder_name=f"{algname}_F{fid}_I{iid}_{dim}D", algorithm_name=f"{algname}")
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
    
    algnames = ["PSO", "ABC", "ABO", "AF", "AIWPSO", "BA", "BOA", "BWO", "CS", "CSA", 
                 "EHO", "FA", "FFOA", "FSO", "GOA", "JS", "KH", "MFO", "MRFO", "NBJS", 
                 "PIO", "PSO", "RPSO", "SAVPSO", "SBO", "SCA", "SFO", "SOS", "SSA", "SSO", 
                 "STOA", "VPSO", "WOA", "AOA", "CEM", "DOA", "GS", "HC", "NDS", "AIG", 
                 "ASO", "BH", "EFO", "EO", "ESA", "GSA", "LSA", "MVO", "WWO", 
                 "WEO", "WDO", "TWO", "TEO", "SA", "BSO", "CI", "ISA", "MVPA", "QSA", 
                 "SSD", "AEO", "AO", "COA", "GCO", "GWO", "HHO", "OSA", 
                 "PPA", "PVS", "RFO", "BSA", "DE", "ES", "FOA", "GA", "GHS", "GOGHS", 
                 "HS", "IHS", "IWO", "NGHS", "RRA", "SGHS"]
    iids = range(1,11)
    
    dims = [2,5,10,20]
    
    args = product(algnames, fids, iids, dims)

    runParallelFunction(run_optimizer, args)
