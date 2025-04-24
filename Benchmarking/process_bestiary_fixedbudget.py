import numpy as np
import pandas as pd
import glob
from functools import partial
from multiprocessing import Pool, cpu_count

import warnings

DATA_FOLDER = ""
CSV_FOLDER = ""

budget_factors = [10,50,100,500,1000,5000,10000]

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
    p = Pool(min(32, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results

def get_auc_table(fname, max_budget_factor = 10000):
    print(fname)
    try:
        dim = int(fname.split('_')[-1][3:-4])
        max_budget = dim * max_budget_factor
        budgets = budget_factors
        print(dim, budgets)
        dt = pd.read_csv(fname, sep=' ', decimal=',')
        dt = dt[dt['raw_y'] != 'raw_y'].astype(float)
        dt['run_id'] = np.cumsum(dt['evaluations'] == 1)
        items = []
        for run in np.unique(dt['run_id']):
            dt_temp = dt[dt['run_id'] == run]
            for budget in budgets:
                items.append([fname, min(dt_temp.query(f"evaluations <= {budget * dim}")['raw_y']), run, budget, budget * dim])
        dt_auc = pd.DataFrame.from_records(items, columns=['fname', 'fx', 'run', 'budget_factor', budget])
        return dt_auc
    except:
        print(f"Failed: {fname}")

def find_files(libname):
    files = glob.glob(f"{DATA_FOLDER}/{libname}/*/*/IOHprofiler_f*.dat")
    return files

def process(fname, libname):
    base = fname.split('/')[-3]
    if libname in base:
        print(base, base[len(libname)+1:])
        base = base[len(libname)+1:]
    try:
        dt_auc = get_auc_table(fname)
        dt_auc.to_csv(f"{CSV_FOLDER}/csv_{libname}/FBUDGET_{base}.csv")
    except:
        print((fname, libname))
        
def merge_files(libname):
    
    for typename in ['FBUDGET']:
        files_FV = glob.glob(f"{CSV_FOLDER}/csv_{libname}/{typename}_*.csv")
        dt_fvs = []
        for fname in files_FV:
            print(fname)
            try:
                base = fname.split('/')[-1]
                algname = base.split('_')[-4]
                if algname == 'base':
                    algname = base.split('_')[-5]
                elif libname == 'mealpy' and (algname == 'PSO' or algname == 'SHADE'):
                    algname = F"{base.split('_')[-5]}{algname}"
                fid = int(base.split('_')[-3][1:])
                iid = int(base.split('_')[-2][1:])
                dim = int(base.split('_')[-1][:-5])
                dt = pd.read_csv(fname, index_col=0)
                dt['algname'] = algname
                dt['fid'] = fid
                dt['iid'] = iid
                dt['dim'] = dim
                dt_fvs.append(dt)
            except:
                print(":(")
        dt_fv = pd.concat(dt_fvs)
        dt_fv.to_csv(f"{CSV_FOLDER}/{typename}_{libname}.csv")
    
    
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    for libname in ['Baselines', 'EVOLOPY', 'mealpy', 'NIAPY', 'OPYTIMIZER']:
        files = find_files(libname)
        runfunc = partial(process, libname=libname)
        runParallelFunction(runfunc, files)
        merge_files(libname)