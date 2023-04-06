'''Problems library
'''

import numpy as np
from algormeter.kernel import *


class Smooth (Kernel):
    def __inizialize__(self, dimension):
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = 0.0
        self.XStart = np.ones(dimension)*2 # (1,1, ...,1)

    def _f1(self, x):
        return np.sum(np.array(x)**2)
    
    def _gf1(self, x):
        return 2.*x

Parab = Smooth

class AbsVal (Kernel):
    def __inizialize__(self, dimension):
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = 0.0
        self.XStart = np.ones(dimension)*2 # (1,1, ...,1)

    def _f1(self, x):
        return np.sum(np.abs(np.array(x)))
    
    def _gf1(self, x):
        return np.sign(x)

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
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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


class CVX1 (Kernel):
    '''Convex 1
    '''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([1,3])
        self.optimumPoint = np.array([1.,1])
        self.optimumValue = 0.0

    def _f1(self, x) :
        return abs(x[0]-1) + 200*np.max([0.,abs(x[0])-x[1]])
    
    def _gf1(self, x):
        return np.array([sign(x[0]-1) + 200*(0 if 0 > abs(x[0])-x[1] else sign(x[0])), 200*(0 if 0 > abs(x[0])-x[1] else -1)])

class DemMal (Kernel):
    '''
        Demyanov, Malozemov function
    '''

    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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
    
class Mifflin1 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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
    
class Mifflin2 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([-1.,-1.])
        self.optimumPoint = np.array([1.,0])
        self.optimumValue = -1.

    def _f11(self, x):
        return x[0]**2 + x[1]**2 - 1

    def _f1(self, x) :
        f11 = self._f11(x)
        return -x[0] + 2*f11 + 1.75*abs(f11)
    
    def _gf1(self, x):
        sf11 = sign(self._f11(x))
        return np.array([3.5*x[0]*sf11 + 4*x[0] -1,
                         x[1]*(3.5*sf11 + 4)])

class LQ (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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

class MAXQ (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 20:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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
    

class MAXL (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 20:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.zeros(dimension)
        for i in range(10):
            self.XStart[i] = i
            self.XStart[i+10] = -i
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = 0.

    def _f1(self, x) :
        return np.max(np.abs(x))
    
    def _gf1(self, x):
        i=np.argmax(np.abs(x))
        v = np.zeros(self.dimension)
        v[i] = np.sign(x[i])
        return v
    
class QL (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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
class CB2 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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
    
class CB3 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
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

import algormeter.libs.data as data
class MaxQuad (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 10:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.zeros(dimension) # alternativo
        self.XStart = np.ones(dimension) # primario
        self.optimumPoint = np.array([-0.1262922, -0.0343347, -0.0068291,  0.0263796,  0.067307,  -0.2783819,0.074205,   0.1385046,  0.0840276,  0.0385804])
        self.optimumValue = -0.8414084 
 
    def _f11(self,x):
        r, c, s = data.mqA.shape
        return [x @ data.mqA[:,:,_] @ x - data.mqB[:,_] @ x for _ in range(s)]
        
    def _f1(self, x):
        return np.max(self._f11(x))
    
    def _gf1(self, x):
        i = np.argmax(self._f11(x))
        return 2*data.mqA[:,:,i] @ x - data.mqB[:,i]
    
      
class Rosenbrock (Kernel):
    '''non convessa'''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = np.array([1,1])
        self.optimumValue = 0.0
        self.XStart = np.array([-1.2,1])

    def _f1(self, x):
        return 100*(x[1]-x[0]**2)**2 +(1-x[0])**2
    
    def _gf1(self, x):
        return np.array([2*(200*x[0]**3 - 200*x[1]*x[0] + x[0] -1), 200*(x[1]-x[0]**2)])
    
class Crescent (Kernel):
    '''non convex'''
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = np.array([0,0])
        self.optimumValue = 0.0
        self.XStart = np.array([-1.5,2])

    def _f11(self, x):
        return x[0]**2 + (x[1] - 1)**2 + x[1] -1
    def _f12(self, x):
        return -x[0]**2 - (x[1] - 1)**2 + x[1] +1
    
    def _f1(self, x):
        return max(self._f11(x),self._f12(x))
    
    def _gf1(self, x):
        if np.argmax([self._f11(x),self._f12(x)]) :
            return np.array([-2*x[0],3-2*x[1]]) 
        return np.array([2*x[0],3-2*x[1]]) 
    
class Rosen (Kernel):
    '''Rosen-Suzuki'''
    def __inizialize__(self, dimension):
        if dimension != 4:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = np.array([0,1,2,-1])
        self.optimumValue = -44.
        self.XStart = np.zeros(4)

    def _f11(self, x):
        return x[0]**2 + x[1]**2 + 2*x[2]**2 + x[3]**2 - 5*x[0] - 5*x[1] - 21*x[2] + 7*x[3] 
    def _f12(self, x):
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[0] - x[1] + x[2] - x[3] -8 
    def _f13(self, x):
        return x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[0] - x[3] -10 
    def _f14(self, x):
        return x[0]**2 + x[1]**2 + x[2]**2  + 2*x[0] - x[1] - x[3] -5 
     
    def _f1(self, x):
        f11 = self._f11(x)
        f12 = self._f12(x)
        f13 = self._f13(x)
        f14 = self._f14(x)
        return max(f11, f11 + 10*f12, f11+10*f13, f11+10*f14)
    
    def _gf1(self, x):
        f11 = self._f11(x)
        f12 = self._f12(x)
        f13 = self._f13(x)
        f14 = self._f14(x)
        match np.argmax([f11, f11 + 10*f12, f11+10*f13,f11+10*f14]):
            case  0:
                return np.array([2*x[0] -5, 2*x[1] -5, 4*x[2] -21, 2*x[3] +7 ])
            case  1:
                return np.array([2*x[0] -5, 2*x[1] -5, 4*x[2] -21, 2*x[3] +7 ]) + \
                        10*np.array([2*x[0] +1, 2*x[1] -1, 2*x[2] +1, 2*x[3] -1 ])
            case  2:
                return np.array([2*x[0] -5, 2*x[1] -5, 4*x[2] -21, 2*x[3] +7 ]) + \
                        10*np.array([2*x[0] -1, 4*x[1], 2*x[2], 4*x[3] -1 ])
            case  3:
                return np.array([2*x[0] -5, 2*x[1] -5, 4*x[2] -21, 2*x[3] +7 ]) + \
                        10*np.array([2*x[0] +2, 2*x[1]-1, 2*x[2], -1 ])

      
class Shor (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 5:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        # self.optimumPoint = np.array([1.1244,0.9795,1.4777,0.9202,1.1243]) # libro
        self.optimumPoint = np.array([1.1243585, 0.9794594, 1.4777118, 0.9202446, 1.1242887])
        self.optimumValue = 22.60016
        self.XStart = np.array([0.,0.,0.,0.,1.]) # f(x*) -> 80  

    def _f11(self,x,i):
        s = 0.
        for j in range(5):
            s = s + data.shB[i]*(x[j]-data.shA[i,j])**2
        return s

    def _f1(self, x):            
        return max([self._f11(x,r) for r in range(10)])
    
    def _gf1(self, x):
        i = np.argmax([self._f11(x,r) for r in range(10) ])
        return np.array([2*data.shB[i]*(x[j]-data.shA[i,j]) for j in range(5)])
    
class Goffin (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 50:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = np.zeros(dimension)
        self.optimumValue = 0.
        self.XStart = np.array([i+1 - 25.5 for i in range(50)])  

    def _f1(self, x):   
        return 50*max(x) - np.sum(x)
    
    def _gf1(self, x):
        i = np.argmax(x)
        rv = -np.ones(self.dimension)
        rv[i]= 49.
        return rv
    
    
class TR48 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 48:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = data.TR48_OP
        self.optimumValue = -638565.
        self.XStart = np.zeros(dimension,dtype=float)  

    def _f1(self, x):   
        return data.TR48_D @ np.max(x - data.TR48_A,axis=1) - data.TR48_S @ x
    
    def _gf1(self, x):
        rv = -data.TR48_S
        for j in range(self.dimension):
            i = np.argmax (x - data.TR48_A[j])
            rv[i] += data.TR48_D[j]
        return rv
    
    
class A48 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 48:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = data.A48_OP
        self.optimumValue = -9870.
        self.XStart = np.zeros(dimension,dtype=float)
        self.D = np.ones(self.dimension)  
        self.S = np.ones(self.dimension)  

    def _f1(self, x):   
        return self.D @ np.max(x - data.TR48_A,axis=1) - self.S @ x
    
    def _gf1(self, x):
        rv = -self.S
        for j in range(self.dimension):
            i = np.argmax (x - data.TR48_A[j])
            rv[i] += self.D[j]
        return rv
    
probList_covx = [
    (Smooth,[50000]),
    (AbsVal,[50000]),
    (DemMal,[2]),
    (CB2,[2]),
    (CB3,[2]),
    (Mifflin1,[2]),
    (Mifflin2,[2]),
    (LQ,[2]),
    (QL,[2]),
    (MAXQ,[20]),
    (MAXL,[20]),
    (MaxQuad,[10]),
    (Rosen,[4]),
    (Shor,[5]),
    (Goffin,[50]),
    (TR48,[48]),
    (A48,[48]),
]

probList_no_covx = [
    (Rosenbrock,[2]),
    (Crescent,[2]),
]

probList_base = [
    (Smooth,[2, 5, 500]),
    (ParAbs,[2,5,10,100]),
    (Acad,[2]),
]



