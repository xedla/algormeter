''' AlgorMeter Kernel
'''
__all__ = ['Kernel']
__author__ = "Pietro d'Alessandro"

import math
import os
from typing import Callable
import numpy as np
from numpy import sign
import time
from algormeter.tools import counter

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

    def f (self, x : np.ndarray) -> np.ndarray :
        return self.f1(x) - self.f2(x)
    def gf (self, x : np.ndarray) -> np.ndarray :
        return self.gf1(x) - self.gf2(x)   
    def _f (self, x : np.ndarray) -> np.ndarray :
        return self._f1(x) - self._f2(x)
    def _gf (self, x : np.ndarray) -> np.ndarray :
        return self._gf1(x) - self._gf2(x)
        
    def f1 (self, x : np.ndarray ) -> np.ndarray :
        return self._cacheCall(x,self._f1,self.F1,self.F1Bit, 'f1')
    def gf1 (self, x : np.ndarray) -> np.ndarray :
        return self._cacheCall(x,self._gf1,self.GF1,self.GF1Bit, 'gf1')
    def f2 (self, x : np.ndarray) -> np.ndarray :
        return self._cacheCall(x,self._f2,self.F2,self.F2Bit, 'f2')
    def gf2 (self, x : np.ndarray) -> np.ndarray :
        return self._cacheCall(x,self._gf2,self.GF2,self.GF2Bit, 'gf2')

    def _f1 (self, x : np.ndarray) -> np.ndarray :
        return  np.array(0.)
    def _gf1 (self, x : np.ndarray) -> np.ndarray :
        return np.zeros(self.dimension)
    def _f2 (self, x : np.ndarray) -> np.ndarray :
        return  np.array(0.)
    def _gf2 (self, x : np.ndarray) -> np.ndarray :
        return np.zeros(self.dimension)

    @property
    def f1Xk(self):
        return self.f1(self.Xk)
    @property
    def f2Xk(self):
        return self.f2(self.Xk)
    @property
    def gf1Xk(self):
        return self.gf1(self.Xk)
    @property
    def gf2Xk(self):
        return self.gf2(self.Xk)
    @property
    def fXk(self):
        return self.f1Xk - self.f2Xk
    @property
    def gfXk(self):
        return self.gf1Xk - self.gf2Xk

## loop
        
    def traceLine(self):
        if not self.trace:
            return 

        fXk = self._f(self.Xk)
        if self.K == 0:
            CB = '\033[47m'
            self.fXkPrev = fXk
        elif self.fXkPrev > fXk:
            CB = '\033[102m' # green
        else:
            CB = '\033[103m' # yellow
        CE = '\033[0m'
        
        if self.K == 0: print()
        if self.isf1_only:
            print(CB,f'{self} k:{self.K},f:{self._f(self.Xk):.3f},x:{self._pp(self.Xk)},gf:{self._pp(self._gf(self.Xk))}',CE)
        else:
            print(CB,f'{self} k:{self.K},f:{self._f(self.Xk):.3f},x:{self._pp(self.Xk)},gf:{self._pp(self._gf(self.Xk))},f1:{self._f1(self.Xk):.3f},gf1:{self._pp(self._gf1(self.Xk))},f2:{self._f2(self.Xk):.3f},gf2:{self._pp(self._gf2(self.Xk))}',CE)

    def recalc(self,x):
        '''Recalc at step k
        '''
        if not (self.Xk == self.Xprev).all():
            self.fXkPrev = self._f(self.Xk)                
            self.Xprev = self.Xk
            # if self.fXkPrev > self.f(self.Xk):
            #     self.fXkPrev = self.f(self.Xk)                
        
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
                if callable(self.stop) and self.stop():
                    self.isFound = True
                    break
                if  time.perf_counter() - self.startTime > self.timeout:
                    self.isTimeout = True
                    break
                self.fXkPrev = self._f(self.Xk) 
        finally:
            self.recalc(self.Xkp1)
            if  self.K <= self.maxiterations or self.stop():
                self.isFound = True
            
            self.XStar = self.Xk

            if self.savedata:
                self.data.resize(self.K,self.dimension+1,refcheck=False)
                label = '' if self.label == '' else self.label + ','
                dir = './npy/'
                if not os.path.isdir(dir):
                    os.mkdir(dir)
                np.save(f'{dir}{label}{repr(self)}',self.data)

            if self.trace:
                print('\n\n')

    def stop(self) -> bool:
        '''return True if experiment must stop. Override it if needed'''
        if np.array_equal(self.Xk, self.Xprev): # if null step 
            return False

        rc = bool(np.isclose(self.fXk,self.fXkPrev,rtol=self.relTol,atol=self.absTol)  
                  or np.allclose (self.gfXk,np.zeros(self.dimension),rtol=self.relTol,atol=self.absTol) )
        return rc

    def isSuccess(self) -> bool:
        '''return True if experiment success. Override it if needed'''
        return  self.isMinimum(self.XStar)

    def stats(self):
        def expStatus():
            if self.isSuccess():
                return 'Success'
            if self.isTimeout:
                return 'Timeout'
            if self.K == self.maxiterations:
                return 'MaxIter'
            return 'Fail'
        counter.disable()
        fxstar = float((self.f(self.XStar))[0].astype(float))
        # fxstar = 1.1
        stat = {"Problem" : str(self),
                "Dim": self.dimension,
                "Status":expStatus(),
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

    def minimize(self,algorithm : Callable, iterations : int = 500, trace : bool = False, dbprint: bool = False, **kargs) -> tuple[bool, np.ndarray , np.ndarray]:
        '''Find  minimum of a problem/function by applying an algorithm developed with algormeter.
            returns (Success, X, f(X))
        '''
        self.maxiterations = iterations
        self.trace = trace
        algorithm(self,**kargs)
        return self.isFound, self.Xk, self.fXk

    def setStartPoint(self, startPoint):
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
            print(CB,f'{repr(self)} at x:{self._pp(x)} -> f:{self._f(x):.3f},gf:{self._pp(self._gf(x))}',CE)
        else:
            print(CB,f'{repr(self)} at x:{self._pp(x)} -> f:{self._f(x):.3f},gf:{self._pp(self._gf(x))},f1:{self._f1(x):.3f},gf1:{self._pp(self._gf1(x))},f2:{self._f2(x):.3f},gf2:{self._pp(self._gf2(x))}',CE)

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
    
# def sign(x):
#     # return np.sign(x)
#     if type(x) == np.ndarray:
#         return 2. * ( x >= 0.) - 1.
#     return 1. if x >= 0. else -1.
