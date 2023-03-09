'''Base Problems library
'''

import numpy as np
from algormeter.tools import counter, dbx
from algormeter.kernel import *
 

class ParAbs (Kernel):
    '''
    _f1: X**2 
    _f2: Abs
    '''
    def __inizialize__(self, dimension):
        self.optimumPoint = np.ones(dimension)*.5
        self.optimumValue = -0.25*dimension
        self.XStart = np.ones(self.dimension)*.1 # (1,1, ...,1)

    def _f1(self, x : list):
        return np.sum(np.array(x)**2)
    
    def _gf1(self, x):
        return 2.*x
    
    def _f2(self,x : list):
        return np.sum(abs(np.array(x)))

    def _gf2(self, x):
        return np.sign(np.array(x))

    def success(self):
        ''' Sono punti di ottimo tutti i punti (x1, ..., xn) dove x(i) = .5 o -.5
        '''
        if np.allclose(abs(self.optimumPoint),abs(self.XStar),rtol=self.relTol,atol=self.absTol):
            self.optimumPoint = self.XStar
            return True
        return False

    
class Acad (Kernel):
    '''
    Academic test problem 
        [-1, -1] global minimum
        [-1, 0],[0, -1],[0, 0] critical and no optimal 
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.optimumPoint = np.array([-1.,-1])
        self.optimumValue = -2.0
        self.randomSet(0.,3) # XStart in [-1.5...1.5, -1.5...1.5,]
        self.XStart = np.ones(self.dimension)*.1 
        self.XStart = np.array([+.7, -1.3])

    def _f1(self, x : list) -> float:
        return 3/2*(x[0]**2 + x[1]**2) + x[0] + x[1]
    
    def _gf1(self, x):
        return np.array([ 3*x[0] + 1 ,3*x[1] + 1])
    
    def _f2(self,x : list) -> float:
        return abs(x[0]) +  abs(x[1]) + 1/2*(x[0]**2 + x[1]**2)

    def _gf2(self, x):
        return np.array([sign(x[0]) + x[0],sign(x[1]) + x[1]])


probList_base = [
    (ParAbs,[2,5,10,100]),
    (Acad,[2]),
]
    

