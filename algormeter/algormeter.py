'''
AlgorMeter Main program 

Created on 9 May 2022

@author: Pietro D'Alessandro

'''
__all__ = ['algorMeter']
__author__ = "Pietro d'Alessandro"

import sys
# from importlib import import_module
import pandas as pd
import numpy as np
from typing import Optional, Callable
import datetime
import os
import __main__
from algormeter.tools import counter, dbx

from typing import Optional, Callable, Any, List, Tuple, Type, Dict, TypeVar
from .kernel import Kernel, sign

Problem = Type[Kernel]
Problems = List[Tuple[Problem,List[int]]]
Algorithm = Callable[[Problem], None]
Algorithms = List[Algorithm]
TuneParameters = Optional[List[Tuple[str,List[Any]]]]

def algorMeter(algorithms : Algorithms, problems : Problems,  iterations : int = 500, timeout : int = 180, 
    tuneParameters : TuneParameters = None,
    runs : int = 1, 
    trace : bool = False, dbprint : bool = False, 
    csv : bool = True, savedata : bool = False,
    absTol : float =1.E-4, relTol : float = 1.E-5,  **kwargs) -> Tuple[pd.DataFrame ,pd.DataFrame | np.ndarray] : 
    '''Benchmark for Optimizer algorithms
        - algorithms: algorithms list. *(algoList_simple is available )* 
        - problems: problems list.   *(probList_base, probList_covx, probList_DCJBKM are available)*
        - tuneParameters = None: see README tuneParameters section 
        - iterations = 500: max iterations number 
        - timeout = 180: time out in seconds
        - runs = 1: see README random section 
        - trace = False: see README trace section 
        - dbprint= False: see README dbprint section 
        - csv = True: write a report in csv format in csv folder
        - savedata = False: save data in data folder
        - absTol =1.E-4, relTol = 1.E-5: tolerance used in numpy allClose and isClose
        - **kwargs: python kwargs propagated to algorithms
    '''

    def algoRun (algorithm, experiment, exceptionOn = False, **kwargs ):
        def checkTunePar(ls):
            if not ls:
                return
            for varName, l in ls:
                if  type(varName) is not str:
                    raise ValueError(f'{varName} invalid name. Must be str')
                if   not len(l):
                    raise ValueError(f'{varName}: empty value list {l}')

        params = {}
        # algModule = import_module(algorithm.__module__)
        algModule = sys.modules[algorithm.__module__]

        def scanParams(list):
            if list:
                varName, ls = list[0]
                for v in ls:
                    try:
                        val=round(v,5)
                        setattr(algModule,varName, val)
                        params[varName] = val
                    except Exception as e: 
                        raise ValueError(f'scanning {varName}:',e)
                    yield from scanParams(list[1:])
            else:
                yield params
        
        def prettyAlgo():
            if algorithm.__module__ == algorithm.__name__:
                return algorithm.__name__
            str = algorithm.__module__ + '.' + algorithm.__name__
            t = str.split('.')
            if len(t)>2:
                str = t[-2] + '.' + t[-1]
            return str.replace('__main__.','')

        iter = 0
        st = dict()
        algoDescr = ''
        stats = []
        
        problem,dims = experiment

        checkTunePar(tuneParameters)

        for dim in dims:
            algoDescr = prettyAlgo()
            print(f'Algorithm:{algoDescr}, Problem:{problem.__name__}, Dimension:{dim} ... running', end= '')
            if tuneParameters:
                print()
            ts = datetime.datetime.now()
            excp = None
            for varStat in scanParams(tuneParameters):
                for _ in range(runs):
                    p = problem (dim, **kwargs)  
                    p.setLabel(algoDescr)
                    excp = None
                    try:
                        if runs > 1:
                            p.isRandomRun = True
                        if tuneParameters:
                            print(problem.__name__,str(varStat),end='\r')
                        algorithm(p, **kwargs)
                    except AssertionError as e:
                        raise e
                    except (ArithmeticError, Exception) as e:
                        excp = e
                        counter.log (str(e), 'Error')
                        if (exceptionOn or dbprint or trace) and excp :
                            raise e
                    finally:
                        if tuneParameters:
                            counter.log (str(varStat), 'TuneParams')
                        st = p.stats()
                        if excp: st['Status'] = 'Error'
                        stats.append(st)
                        iter = p.K

            usedtime = datetime.datetime.now() - ts  
            msg = ': ' + str(excp) if excp else ''
            if tuneParameters:
                print('\ntime:',usedtime)
            else:
                print('. Done. time:',usedtime, ' iterations:',iter, ' status:', st['Status'], msg)

        s = pd.DataFrame(stats)
        s.insert(2, 'Algorithm', algoDescr)
        pd.options.display.width = 0
        return s


    np.set_printoptions(precision=7)
    dbx._DBPRINT = dbprint

    dfl = []
    for exp in problems:
        for alg in algorithms:
            dfl.append(algoRun(experiment=exp,algorithm=alg,tuneParameters = tuneParameters,
                iterations=iterations, timeout = timeout, runs = runs, trace = trace, csv = csv, dbprint=dbprint,
                absTol = absTol, relTol = relTol, savedata = savedata, **kwargs ))

    if not dfl:
        print('Empty result')
        exit()
        
    df = pd.concat(dfl).sort_values(by=['Problem','Dim','Algorithm']) if len(dfl) > 1 else dfl[0]
    df = df.reset_index(drop=True)
    df = df.astype({'Dim':int, 'Delta':float}) 


    if csv:
        if hasattr(__main__, '__file__'):
            expName = __main__.__file__.split('/')[-1].split('.')[0]
        else:
            expName = 'jupyter'
        now  = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') 
        dir = './csv/'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        df.to_csv(f'{dir}{now} {expName}.csv',index=False)

    try:
        pv = np.round(pd.pivot_table(df, values='Problem',index='Algorithm',columns='Status',aggfunc='count',margins = True,fill_value=0),2)
    except Exception as e:
        pv = pd.DataFrame()
 
    return df, pv
