'''Coax Problems library
'''

import numpy as np
from algormeter.tools import counter, dbx
from algormeter.kernel import *

   
class Parab (Kernel):
    '''
    _f1: Parabola with min in 0
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        self.optimumPoint = np.zeros(self.dimension)
        self.optimumValue = 0.0
        self.XStart = np.ones(self.dimension)*2 # (1,1, ...,1)

    def _f1(self, x):
        return np.sum(np.array(x)**2)
    
    def _gf1(self, x):
        return 2.*x
    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)

class DemMol (Kernel):
    '''
    _f1: Demyanov, Molozemov function
    '''

    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.optimumPoint = np.array([0.,-3.])
        self.optimumValue = -3.0
        self.XStart = np.array([1.,1.])
        
    def _f1(self, x) :
        return np.max([5*x[0]+x[1],-5*x[0]+x[1],x[0]**2+x[1]**2+4*x[1]])
    
    def _gf1(self, x):
        idx = np.argmax([5*x[0]+x[1],-5*x[0]+x[1],x[0]**2+x[1]**2+4*x[1]])
        if idx == 0:
            return np.array([5.,1.])
        elif idx == 1:
            return np.array([-5.,1.])
        else:
            return np.array([2.*x[0],2.*x[1] + 4.])
    
    def _f2(self,x ) :
        return 0

    def _gf2(self, x):
        return np.zeros(self.dimension)

class Mifflin (Kernel):
    '''
    _f1: Mifflin
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.array([.8,.6])
        self.optimumPoint = np.array([1.,0])
        self.optimumValue = -1.

    def _f11(self, x):
        return x[0]**2 + x[1]**2 - 1

    def _f1(self, x) :
        return -x[0]+20*max(self._f11(x), 0.)
    
    def _gf1(self, x):
        if self._f11(x) > 0.:
            return np.array([-1. + 40*x[0],40*x[1]])
        else:
            return np.array([-1.,0.])
    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)

class LQ (Kernel):
    '''
    _f1: LQ
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.array([-.5,-.5])
        sq2 = np.sqrt(2)
        self.optimumPoint = np.array([1./sq2,1./sq2])
        self.optimumValue = -sq2

    def _f11(self, x):
        return -x[0] - x[1]

    def _f12(self, x):
        return -x[0] - x[1] + x[0]**2 + x[1]**2 - 1

    def _f1(self, x) :
        return max(self._f11(x), self._f12(x))
    
    def _gf1(self, x):
        if self._f11(x) > self._f12(x):
            return np.array([-1. , -1.])
        else:
            return np.array([-1. + 2*x[0], -1. + 2*x[1]])
    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)

class MAXQ (Kernel):
    '''
    _f1: MAXQ
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        if dimension != 20:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.zeros(dimension)
        for i in range(10):
            self.XStart[i+10] = -i
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = 0.

    def _f1(self, x) :
        return np.max(x**2)
    
    def _gf1(self, x):
        i=np.argmax(x**2)
        v = np.zeros(self.dimension)
        v[i] = 2*x[i]
        return v
    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)


class QL (Kernel):
    '''
    _f1: QL
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.array([-1.,5.])
        self.optimumPoint = np.array([1.2,2.4])
        self.optimumValue = 7.2

    def _f11(self, x):
        return x[0]**2 + x[1]**2

    def _f12(self, x):
        return x[0]**2 + x[1]**2 +10*(-4*x[0] - x[1] + 4.)

    def _f13(self, x):
        return x[0]**2 + x[1]**2 +10*(-x[0] - 2*x[1] + 6.)

    def _f1(self, x) :
        return max(self._f11(x), self._f12(x), self._f13(x))
    
    def _gf1(self, x):
        match np.argmax([self._f11(x), self._f12(x), self._f13(x)]):
            case  0:
                return np.array([2*x[0],2*x[1]])
            case  1:
                return np.array([2*x[0]-40.,2*x[1]-10.])
            case  2:
                return np.array([2*x[0]-10.,2*x[1]-20.])
    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)

class CB2 (Kernel):
    '''
    _f1: CB2
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.array([1.,-.1])
        self.optimumPoint = np.array([1.1392286, 0.899365])
        # self.optimumValue = 1.9522245 # value on paper. Float not double like python?
        self.optimumValue = 1.9523248

    def _f11(self, x):
        return x[0]**2 + x[1]**4

    def _f12(self, x):
        return (2. - x[0])**2 + (2. - x[1])**2

    def _f13(self, x):
        return 2*np.exp(-x[0]+x[1])

    def _f1(self, x) :
        return max(self._f11(x), self._f12(x), self._f13(x))
    
    def _gf1(self, x):
        match np.argmax([self._f11(x), self._f12(x), self._f13(x)]):
            case  0:
                return np.array([2*x[0],4*x[1]**3])
            case  1:
                return np.array([-4+2*x[0],-4+2*x[1]])
            case  2:
                return np.array([-2*np.exp(-x[0]+x[1]),2*np.exp(-x[0]+x[1])])
    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)


class CB3 (Kernel):
    '''
    _f1: CB3
    _f2: zero
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.array([2.,2.])
        self.optimumPoint = np.array([1., 1.])
        self.optimumValue = 2.

    def _f11(self, x):
        return x[0]**4 + x[1]**2
    def _f12(self, x):
        return (2. - x[0])**2 + (2. - x[1])**2
    def _f13(self, x):
        return 2*np.exp(-x[0]+x[1])

    def _f1(self, x) :
        return max(self._f11(x), self._f12(x), self._f13(x))
    
    def _gf1(self, x):
        match np.argmax([self._f11(x), self._f12(x), self._f13(x)]):
            case  0:
                return np.array([4*x[0]**3,2*x[1]])
            case  1:
                return np.array([-4+2*x[0],-4+2*x[1]])
            case  2:
                return np.array([-2*np.exp(-x[0]+x[1]),2*np.exp(-x[0]+x[1])])

    
    def _f2(self,x) :
        return 0.

    def _gf2(self, x):
        return np.zeros(self.dimension)

class CVX1 (Kernel):
    '''Convex 1
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'Dimension {dimension} not supported ')
        self.XStart = np.array([-1.,-2.])
        self.optimumPoint = np.array([1.,1])
        self.optimumValue = 0.0

    def _f1(self, x) :
        return abs(x[0]-1) + 200*np.max([0.,abs(x[0])-x[1]])
    
    def _gf1(self, x):
        return np.array([sign(x[0]-1) + 200*(0 if 0 > abs(x[0])-x[1] else sign(x[0])), 200*(0 if 0 > abs(x[0])-x[1] else -1)])

    def _f2(self,x ) :
        return 0.

    def _gf2(self, x):
        return np.array([0, 0])

probList_coax = [
    # (Parab,[2, 5, 20]),
    (DemMol,[2]),
    (Mifflin,[2]),
    (LQ,[2]),
    (MAXQ,[20]),
    (QL,[2]),
    (CB2,[2]),
    (CB3,[2]),
]

    

