''' AlgoMeter Kernel
'''
__all__ = ['Kernel']
__version__ = '0.9'
__author__ = "Pietro d'Alessandro"

import math
from typing import Optional, Callable
import numpy as np
import timeit  as tm
from algormeter.tools import counter, dbx

class Kernel:
    CACHESIZE = 128
    SAVEDATABUFFERSIZE = 1000

    def __init__ (self, dimension : int =2) :
        self.dimension = dimension 
        self.initCache(dimension)
        self.isRandomRun = False
        self.__inizialize__(dimension)
        self.randomSet() # default random run param
        self.label = ''
        self.Success = False
        self.config() 
        self.K = -1
        self.Xk = self.XStart
        self.clearCache()
        counter.reset()
        self.recalc(self.XStart)

    def __inizialize__(self, dimension : int):
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = 0.0
        self.XStart = np.ones(dimension)

    def config (self, iterations : int =500, trace : bool = False, savedata : bool = False,
                csv :bool = False, relTol : float = 1.E-5, absTol : float = 1.E-8, **kwargs) -> None :
        '''configure with default value'''
        self.trace = trace
        self.csv = csv
        self.maxiterations = iterations
        self.relTol = relTol 
        self.absTol = absTol 
        self.savedata = savedata
        self.Xk = self.XStart

        if self.savedata is True:
            self.data = np.zeros([Kernel.SAVEDATABUFFERSIZE,self.dimension+1]) # +1 per fx
            self.X = self.data[:,:-1] 
            self.Y = self.data[:,-1] 

    def initCache(self,dim,cachesize = CACHESIZE):
        self.__cache = np.zeros((3*dim+3)*cachesize,dtype=float) # x,gf1,gf2,f1,f2,flags
        self.__cache = self.__cache.reshape(cachesize,-1)
        self.ci = 0 # cache index, prossima row da utilizzare
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

    def _cacheCall(self, x, func, storage, mask, label):
        x = np.array(x)
        i, flags = self.XFinder(x)
        # print('x:',x,'idx:',i,'flags:',flags,'label:', label)

        if flags and (flags& mask): # found 
                # counter.up(label, cls='cache')
                return storage[i]

        counter.up(label) # not in cache 
        if flags: # but x is present in cache
            newMask = int(self.FLAGS[i]) | mask
        else: # x is not present in cache
            newMask = mask
            i = self.ci
            self.ci = (self.ci +1) % self.cachesize

        self.XC[i] = x 
        storage[i] = func(x)
        self.FLAGS[i] =  newMask
        return storage[i]

    def XFinder(self,x):
        idx = np.where(np.all(self.XC==x,axis=1))[0]
        r = int(idx[0]) if idx.size > 0 else None 
        flags = int(self.FLAGS[r]) if r is not None else None 
        return r, flags # flags is None if x not exist in cache

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
        
    
    def traceLine(self):
        if self.trace:
            CB = '\033[102m'
            CE = '\033[0m'
            if self.K == 0: print()
            print(CB,f'{repr(self)} k:{self.K},f:{self._f(self.Xk):.3f},x:{self._pp(self.Xk)},gf:{self._pp(self._gf(self.Xk))},f1:{self._f1(self.Xk):.3f},gf1:{self._pp(self._gf1(self.Xk))},f2:{self._f2(self.Xk):.3f},gf2:{self._pp(self._gf2(self.Xk))}',CE)

      
    def nullStepIsStop(self):
        '''Declare Null Step. Do I stop for maxiteration?'''
        self.recalc(self.Xk)
        if self.K >= self.maxiterations-1:
            return True
        return False

    def recalc(self,x):
        '''Recalc at step k
        '''
        self.K +=1 # start from -1
        self.Xk = x
        
        self.Xkp1 = x
        k = self.K

        if self.savedata:
            # resize data se necessario
            r,_ = self.data.shape
            if k == r:
                self.data.resize(k+Kernel.SAVEDATABUFFERSIZE,self.dimension+1,refcheck=False)
                self.X = self.data[:,:-1] 
                self.Y = self.data[:,-1] 

            self.X[k] = self.Xk
            self.Y[k] = self.fXk
        self.traceLine()

    def loop(self):
        self.startTime = tm.default_timer()
        counter.reset()
        if self.isRandomRun:
            self.randomStartPoint()
        self.kXMin = 0
        self.fXMin = math.inf 
        self.XMin = self.XStart
        self.Success = False
        self.XStar = self.XStart
        self.Xk = self.XStart
        # self.Xkp1 = self.XStart
        self.fXkPrev = math.inf
        self.K = -1 # recalc inc K
        self.recalc(self.Xk)
        
        while self.K < self.maxiterations-1:
            yield self.K
            self.recalc(self.Xkp1)
            if self.fXk < self.fXMin:
                self.XMin = self.Xkp1
                self.kXMin = self.K
                self.fXMin = self.fXk
            if self.isHalt():
                self.Success = True
                break
            self.fXkPrev = self.fXk
        
        self.XStar = self.Xk

        if self.savedata:
            self.data.resize(self.K,self.dimension+1,refcheck=False)
            label = '' if self.label == '' else self.label + ','
            np.save(f'./npy/{label}{repr(self)}',self.data)

        if self.trace:
            print('\n\n')

    def isSuccess(self) -> bool:
        '''return True if experiment success. Reassign it if needed'''
        return self.Success and bool(np.isclose(self.f(self.XStar), self.optimumValue,atol=self.absTol, rtol= self.relTol)) 

    def isHalt(self) -> bool:
        '''return True if experiment must stop. Reassign it if needed'''
        return bool(np.isclose(self.fXk,self.fXkPrev,rtol=self.relTol,atol=self.absTol)  or \
                np.allclose (self.gfXk,np.zeros(self.dimension),rtol=self.relTol,atol=self.absTol) )

    def stats(self):
        def success2Status():
            if self.isSuccess():
                return 'Success'
            return 'Fail'
        counter.disable()
        stat = {"Problem" : str(self),
                "Dim": self.dimension,
                "Status":success2Status(),
                "Iterations": int(self.K),
                "f(XStar)": f'{float(self.f(self.XStar)):.7G}',
                "f(BKXStar)":  f'{self.optimumValue:.7G}',
                'Delta': f'{(abs(self.optimumValue-float(self.f(self.XStar)))):.0E}',
                "Seconds" :f'{(tm.default_timer() - self.startTime):.2f}',
                "Start point": self._pp(self.XStart),
                # "Distance": round(np.linalg.norm(self.optimumPoint - self.XStar),3),
                "XStar": self._pp(self.XStar),
                "BKXStar":  self._pp(self.optimumPoint),
                # "MinX": self._pp(self.XMin),
                # "f(MinX)": self.fXMin,
            }
        stat.update(counter.report())
        counter.enable()
        return stat

    def minimize(self,algorithm : Callable, iterations : int = 500, trace : bool = False, **kargs) -> tuple[bool, np.ndarray , np.ndarray]:
        '''Find  minimum of a problem/function by applying an algorithm developed with algormeter.
            returns (Success, X, f(X))
        '''
        self.maxiterations = iterations
        self.trace = trace
        algorithm(self,**kargs)
        return self.Success, self.Xk, self.fXk

    def setStartPoint(self, startPoint):
        if (len(startPoint) != self.dimension):
            raise ValueError('bad dimension')
        self.XStart = np.array(startPoint)

    def randomSet(self, center = 0., size =1.):
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
        print(CB,f'{repr(self)} at x:{self._pp(x)} -> f:{self._f(x):.3f},gf:{self._pp(self._gf(x))},f1:{self._f1(x):.3f},gf1:{self._pp(self._gf1(x))},f2:{self._f2(x):.3f},gf2:{self._pp(self._gf2(x))}',CE)

    def __str__ (self):
        return self.__class__.__name__  

    def __repr__ (self):
        return self.__class__.__name__ + "-" +str(self.dimension)
  
    def setLabel(self,label):
        self.label = label
    
    @staticmethod
    def _pp(s):
        r = np.array2string(s,precision=4,threshold=4)
        return r


def sign(x):
    # return np.sign(x)
    if type(x) == np.ndarray:
        return 2. * ( x >= 0.) - 1.
    return 1. if x >= 0. else -1.
