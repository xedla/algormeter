''' AlgorMeter Kernel
'''
__all__ = ['Kernel']
__author__ = "Pietro d'Alessandro"

import math
import os
from typing import Callable, Literal, Tuple
import numpy.typing as npt
import numpy as np
import time
from algormeter.tools import counter, dbx
from numpy import sign

import warnings

Array1D = npt.NDArray[np.float64]

    # https://gist.github.com/vratiu/9780109 colori
class Color:
    Black='\033[1;40m' 
    Red='\033[1;41m'   
    Green='\033[1;42m' 
    Yellow='\033[1;43m'
    Blue='\033[1;44m'  
    Purple='\033[1;45m'
    Cyan='\033[1;46m'  
    White='\033[1;47m' 
    Clean='\033[0m' 
    


class AlgorMeterWarning(Warning):
    pass

class Kernel:

## cache 
    CACHESIZE = 128
    SAVEDATABUFFERSIZE = 1000

    def initCache(self,dim,cachesize = CACHESIZE):
        self.__cache = np.zeros((3*dim+3)*cachesize,dtype=float) # x,gf1,gf2,f1,f2,flags
        self.__cache = self.__cache.reshape(cachesize,-1)
        self.dim = dim
        self.cachesize = cachesize
        self.XC = self.__cache[:,0:self.dim]
        self.GF1 = self.__cache[:,self.dim:2*self.dim]
        self.GF2 = self.__cache[:,2*self.dim:3*self.dim]
        self.F1 = self.__cache[:,3*self.dim:3*self.dim+1]
        self.F2 = self.__cache[:,3*self.dim+1:3*self.dim+2]
        self.FLAGS = self.__cache[:,3*self.dim+2:3*self.dim+3]
        self.GF1Bit = 0x01
        self.GF2Bit = 0x02
        self.F1Bit = 0x04
        self.F2Bit = 0x08

    def clearCache(self):
        self.__cache = np.zeros((3*self.dim+3)*self.cachesize,dtype=float) # x,gf1,gf2,f1,f2,flags
        self.norms = -1*np.ones(Kernel.CACHESIZE) # norms as cache index
        self.lru = np.zeros(Kernel.CACHESIZE,dtype=int) # last recent used
        self.ccc = 0

    def _cacheCall(self, x, func, storage, mask, label):
        i, flags = self.XFinder(x)
        self.ccc += 1
        
        if flags and (flags & mask): # found 
                self.lru[i] = self.ccc 
                return storage[i]

        counter.up(label) # not in cache, count it
        if flags: # but x is present in cache
            self.lru[i] = self.ccc 
            newMask = self.FLAGS[i].astype(int) | mask
        else: # x is not present in cache
            newMask = mask
            i = np.argmin(self.lru)
            self.XC[i] = x 
            self.norms[i] = np.linalg.norm(x)

        storage[i] = func(x)
        self.FLAGS[i] =  newMask
        self.lru[i] = self.ccc 
        return storage[i]

    def XFinder(self,x):
        r , flags = None, None
        
        nx = np.linalg.norm(x)
   
        for i in np.where(self.norms==nx)[0]:
            if np.all(self.XC[i]==x):
                r = i
                flags = self.FLAGS[r].astype(int) 
                break
            
        return r, flags # flags is None if x not exist in cache

## problem interface
    def __init__ (self, dimension : int , iterations : int =500, timeout : int = 180, trace : bool = False, savedata : bool = False,
                csv :bool = False, relTol : float = 1.E-5, absTol : float = 1.E-8, **kwargs) :
        self.dimension = dimension 
        self.initCache(dimension)
        self.isTimeout = False
        self.isRandomRun = False
        self.startTime = time.perf_counter()
        self.XStart :Array1D = np.zeros(dimension)

        self.__inizialize__(dimension)
        '''configure with default value'''
        self.trace = trace
        self.csv = csv
        self.timeout = timeout
        self.maxiterations = iterations
        self.relTol = relTol 
        self.absTol = absTol 
        self.savedata = savedata
        self.Xk = self.XStart

        self.randomSet() # default random run param
        self.label = ''
        self.isf1_only = 'Kernel' in self._f2.__qualname__ and 'Kernel' in self._gf2.__qualname__
        self.K = -1
        self.Xk = self.XStart
        self.XStar = np.empty(dimension)
        self.Xprev = self.Xk
        self.fXkPrev = math.inf
        self.isFound = False
        if self.savedata is True:
            self.data = np.zeros([Kernel.SAVEDATABUFFERSIZE,self.dimension+1]) # +1 per fx
            self.X = self.data[:,:-1] 
            self.Y = self.data[:,-1] 
        self.clearCache()
        counter.reset()
        self.recalc(self.XStart)


    def __inizialize__(self, dimension : int):
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = math.inf
        self.XStart = np.ones(dimension)

    def config (self, iterations : int =500, timeout : int = 180, trace : bool = False, savedata : bool = False,
                csv :bool = False, relTol : float = 1.E-5, absTol : float = 1.E-8, **kwargs) -> None :
        '''configure with default value'''
        self.trace = trace
        self.csv = csv
        self.timeout = timeout
        self.maxiterations = iterations
        self.relTol = relTol 
        self.absTol = absTol 
        self.savedata = savedata
        self.Xk = self.XStart

        if self.savedata is True:
            self.data = np.zeros([Kernel.SAVEDATABUFFERSIZE,self.dimension+1]) # +1 per fx
            self.X = self.data[:,:-1] 
            self.Y = self.data[:,-1] 
    
    def isMinimum(self, x : np.ndarray) -> bool:
        return math.isclose(self._f(x),self.optimumValue, rel_tol = self.relTol, abs_tol = self.absTol)

    def f (self, x : Array1D) -> float :
        return self.f1(x) - self.f2(x)
    def gf (self, x : Array1D) -> Array1D :
        return self.gf1(x) - self.gf2(x)   
    def _f (self, x : Array1D) -> float :
        return self._f1(x) - self._f2(x)
    def _gf (self, x : Array1D) -> Array1D :
        return self._gf1(x) - self._gf2(x)
        
    def f1 (self, x : Array1D ) -> float :
        return float(self._cacheCall(x,self._f1,self.F1,self.F1Bit, 'f1')[0])
    def gf1 (self, x : Array1D) -> Array1D :
        return self._cacheCall(x,self._gf1,self.GF1,self.GF1Bit, 'gf1')
    def f2 (self, x : Array1D) -> float :
        return float(self._cacheCall(x,self._f2,self.F2,self.F2Bit, 'f2')[0])
    def gf2 (self, x : Array1D) -> Array1D :
        return self._cacheCall(x,self._gf2,self.GF2,self.GF2Bit, 'gf2')

    def _f1 (self, x : Array1D) -> float :
        return  0.
    def _gf1 (self, x : Array1D) -> Array1D :
        return np.zeros(self.dimension)
    def _f2 (self, x : Array1D) -> float :
        return  0.
    def _gf2 (self, x : Array1D) -> Array1D :
        return np.zeros(self.dimension)

    @property
    def f1Xk(self) -> float:
        return self.f1(self.Xk)
    @property
    def f2Xk(self) -> float:
        return self.f2(self.Xk)
    @property
    def gf1Xk(self) -> Array1D:
        return self.gf1(self.Xk)
    @property
    def gf2Xk(self) -> Array1D:
        return self.gf2(self.Xk)
    @property
    def fXk(self) -> float:
        return self.f1Xk - self.f2Xk
    @property
    def gfXk(self) -> Array1D:
        return self.gf1Xk - self.gf2Xk

    def traceLine(self):
        if not self.trace or self.K < 0:
            return 

        fXk = self._f(self.Xk)

        if self.K == 0:
            self.fXkPrev = fXk
            CB = Color.White #   first iteration
        elif self.isFound:
            CB = Color.Blue #  minimum found
        elif self.fXkPrev > fXk:
            CB = Color.Green # decrease
        else:
            CB = Color.Red # not decrease
        CE = Color.Clean
        
        if self.K == 0: print()
        if self.isf1_only:
            print(CB,f'{self} k:{self.K},f:{self._f(self.Xk)},x:{self._pp(self.Xk)},gf:{self._pp(self._gf(self.Xk))}',CE)
        else:
            print(CB,f'{self} k:{self.K},f:{self._f(self.Xk)},x:{self._pp(self.Xk)},gf:{self._pp(self._gf(self.Xk))},f1:{self._f1(self.Xk)},gf1:{self._pp(self._gf1(self.Xk))},f2:{self._f2(self.Xk)},gf2:{self._pp(self._gf2(self.Xk))}',CE)


    def recalc(self,x):
        '''Recalc at step k
        '''
        if not (self.Xk == self.Xprev).all():
            self.fXkPrev = self._f(self.Xk)                
            self.Xprev = self.Xk
            fxk = self._f(self.Xk)
            if (self.K > 1 and self.fXkPrev < fxk):
                warnings.warn(f'The objective function f has increased in value {self.fXkPrev} -> {fxk} at iteration {self.K}', AlgorMeterWarning)
        
        self.Xk = x
        self.Xkp1 = x

        self.traceLine()

        if self.savedata:
            k = self.K
            # resize data se necessario
            r,_ = self.data.shape
            if k == r:
                self.data.resize(k+Kernel.SAVEDATABUFFERSIZE,self.dimension+1,refcheck=False)
                self.X = self.data[:,:-1] 
                self.Y = self.data[:,-1] 

            self.X[k] = self.Xk
            self.Y[k] = self._f(self.Xk)

    def loop(self):
        self.startTime = time.perf_counter()
        if self.isRandomRun:
            self.randomStartPoint()
        self.isFound = False
        self.isTimeout = False
        self.XStar = self.XStart
        self.Xk = self.XStart
        self.K = 0
        self.recalc(self.Xk)
        
        try:
            for self.K in range(1,self.maxiterations +1):
                yield self.K
                self.recalc(self.Xkp1)
                self.isFound = self.isHalt()
                if self.isFound:
                    break
                if  self.K >= self.maxiterations:
                    self.isFound = False
                    break
                if  time.perf_counter() - self.startTime > self.timeout:
                    self.isTimeout = True
                    break
                self.fXkPrev = self._f(self.Xk) 
        finally:
            self.isFound = self.isHalt()
            self.XStar = self.Xk
            self.recalc(self.XStar)

            if self.savedata:
                self.data.resize(self.K,self.dimension+1,refcheck=False)
                label = '' if self.label == '' else self.label + ','
                dir = './npy/'
                if not os.path.isdir(dir):
                    os.mkdir(dir)
                np.save(f'{dir}{label}{repr(self)}',self.data)

            if self.trace:
                print('\n')

    def isHalt(self) -> bool:
        '''return True if experiment must stop. Override it if needed'''
        # if np.array_equal(self.Xk, self.Xprev): # if null step 
        #     return False

        rc = bool(np.isclose(self.fXk,self.fXkPrev,rtol=self.relTol,atol=self.absTol)  
                  or np.allclose (self.gfXk,np.zeros(self.dimension),rtol=self.relTol,atol=self.absTol) )
        return rc

    def isSuccess(self) -> bool:
        '''return True if experiment success. Override it if needed'''
        return  self.isMinimum(self.XStar)

    def expStatus(self) -> Literal['Success', 'Timeout', 'MaxIter', 'Fail']:
        if self.isSuccess():
            return 'Success'
        if self.isTimeout:
            return 'Timeout'
        if self.K == self.maxiterations:
            return 'MaxIter'
        return 'Fail'
    
    def stats(self):
        counter.disable()
        fxstar = float((self.f(self.XStar)))
        # fxstar = 1.1
        stat = {"Problem" : str(self),
                "Dim": self.dimension,
                "Status":self.expStatus(),
                "Iterations": int(self.K),
                "f(XStar)": f'{fxstar:.7G}',
                "f(BKXStar)":  f'{self.optimumValue:.7G}',
                'Delta': f'{(abs(self.optimumValue-fxstar)):.1E}',
                "Seconds" :f'{(time.perf_counter() - self.startTime):.4f}',
                "XStar": self._pp(self.XStar),
                "BKXStar":  self._pp(self.optimumPoint),
                "Start point": self._pp(self.XStart),
            }
        stat.update(counter.report())
        counter.enable()
        return stat

    def minimize(self,algorithm : Callable,*, iterations : int = 500, 
                 absTol : float =1.E-4, relTol : float = 1.E-5,
                 startPoint: Array1D | None = None,
                 trace : bool = False, dbprint: bool = False, **kargs) -> \
            Tuple[Literal['Success', 'Timeout', 'MaxIter', 'Fail'], Array1D , float]:
        '''Find  minimum of a problem/function by applying an algorithm developed with algormeter.
            returns (Success, X, f(X))
        '''
        def status():
            if self.isFound:
                return 'Success'
            if self.isTimeout:
                return 'Timeout'
            if self.K == self.maxiterations:
                return 'MaxIter'
            return 'Fail'
        self.maxiterations = iterations
        self.trace = trace
        dbx.dbON(dbprint)
        self.absTol = absTol
        self.relTol = relTol
        self.isFound = False
        if startPoint is not None:
            self.XStart = startPoint 
        algorithm(self, **kargs)
        return status(), self.Xk, self.fXk

    def setStartPoint(self, startPoint :Array1D):
        if (len(startPoint) != self.dimension):
            raise ValueError('bad dimension')
        self.XStart = np.array(startPoint)

    def randomSet(self, center:float = 0., size: float = 1.) -> None:
        ''' set random run center and size'''
        self.startRandRect = np.array([center - size, center + size])

    def randomStartPoint(self):
        ''' set random start point
        '''
        sr = self.startRandRect 
        self.XStart = np.random.rand(self.dimension)*(sr[1] - sr[0]) + np.ones(self.dimension)*sr[0]
        self.Xk = self.XStart

    def __call__(self, x):
        x = np.array(x)
        CB = '\033[102m'
        CE = '\033[0m'
        if self.isf1_only:
            print(CB,f'{str(self)} at x:{self._pp(x)} -> f:{self._f(x)},gf:{self._pp(self._gf(x))}',CE)
        else:
            print(CB,f'{str(self)} at x:{self._pp(x)} -> f:{self._f(x)},gf:{self._pp(self._gf(x))},f1:{self._f1(x)},gf1:{self._pp(self._gf1(x))},f2:{self._f2(x)},gf2:{self._pp(self._gf2(x))}',CE)

    def __str__ (self):
        return self.__class__.__name__

    def __repr__ (self):
        return super(Kernel, self).__repr__() + " Dimension:" +str(self.dimension)
  
    def setLabel(self,label):
        self.label = label
    
    @staticmethod
    def _pp(s):
        r = np.array2string(s,precision=4,threshold=4)
        return r
    
# def sign(x) -> bool:
#     if type(x) == Array1D:
#         return bool(np.sign(x))
#     return True if float(x) >= 0. else False
