'''
AlgoMeter Main program 

Created on 9 May 2022

@author: Pietro D'Alessandro

'''
__all__ = ['algorMeter']
__version__ = '0.9'
__author__ = "Pietro d'Alessandro"

import pandas as pd
import numpy as np
from typing import Optional, Callable
import datetime
import os
import __main__
from algormeter.tools import counter, dbx
class TunePar:
    ...


def algorMeter(algorithms : list[Callable], problems : list[tuple[Callable,list[int]]], tuneParameters : Optional[list] = None, iterations : int = 500, 
    runs : int = 1, trace : bool = False, dbprint : bool = False, csv : bool = True, savedata : bool = False,
     absTol : float =1.E-4, relTol : float = 1.E-5,  **kwargs) -> tuple[pd.DataFrame ,pd.DataFrame | np.ndarray] : 
    '''Benchmark environment for Optimizer algorithms
        - algorithms: algorithms list. *(algoList_simple is available )* 
        - problems: problem list. See problems list in example4.py for syntax.   *(probList_base, probList_coax, probList_DCJBKM are available)*
        - tuneParameters = None: see tuneParameters section 
        - iterations = 500: max iterations number 
        - runs = 1: see random section 
        - trace = False: see trace section 
        - dbprint= False: see dbprint section 
        - csv = True: write a report in csv format in csv folder
        - savedata = False: save data in data folder
        - absTol =1.E-4, relTol = 1.E-5: tolerance used in numpy allClose and isClose
        - **kwargs: python kwargs propagated to algorithms
    '''

    def algoRun (algorithm, experiment, **kwargs ):
        def checkTunePar(ls):
            if not ls:
                return
            for varName, l in ls:
                if  'TunePar' not in varName:
                    raise ValueError(f'{varName} invalid name. Must be TunePar.<someparam>')
                if   not len(l):
                    raise ValueError(f'{varName}: empty value list {l}')

        def scanParams(list):
            def paramStatus():
                vs = {}
                for k,e  in vars(TunePar).items():
                    if not k.startswith('__'):
                        vs[k] = e
                return vs

            if list:
                varName, ls = list[0]
                for v in ls:
                    try:
                        exec(f'{varName}=round({v},3)')
                    except Exception as e: 
                        raise ValueError(f'parameter {varName}:',e)
                    yield from scanParams(list[1:])
            else:
                yield paramStatus()
        
        def prettyAlgo():
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
            ts = datetime.datetime.now()
            excp = None
            for varStat in scanParams(tuneParameters):
                for _ in range(runs):
                    p = problem (dimension=dim)  
                    p.config(**kwargs)
                    p.setLabel(algoDescr)
                    excp = None
                    try:
                        if runs > 1:
                            p.isRandomRun = True
                        algorithm(p, **kwargs)
                        if tuneParameters:
                            counter.log (str(varStat), 'TunePar')
                    except AssertionError as e:
                        raise e
                    except (ArithmeticError, Exception) as e:
                        excp = e
                        if (dbprint or trace) and excp :
                            raise e
                    finally:
                        st = p.stats()
                        if excp: st['Status'] = 'Error'
                        stats.append(st)
                        iter = p.K

            usedtime = datetime.datetime.now() - ts  
            msg = ': ' + str(excp) if excp else ''
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
                iterations=iterations, runs = runs, trace = trace, csv = csv, dbprint=dbprint,
                absTol = absTol, relTol = relTol, savedata = savedata ))

    if not dfl:
        print('Empty result')
        exit()
        
    df = pd.concat(dfl).sort_values(by=['Problem','Dim','Algorithm']) if len(dfl) > 1 else dfl[0]
    df = df.reset_index(drop=True)
    df['Delta'] = pd.to_numeric(df['Delta'])

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
