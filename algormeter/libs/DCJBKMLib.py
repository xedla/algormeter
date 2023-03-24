'''Difference of Convex functions Problems library
Kaisa Joki, Adil M. Bagirov, Napsu Karmitsa & Marko M. Mäkelä 
    https://link.springer.com/article/10.1007/s10898-016-0488-3
'''

import numpy as np
from algormeter.kernel import *

class JB01 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.optimumPoint = np.array([1.,1.])
        self.optimumValue = 2.0
        self.XStart = np.array([2.,2.])

    def f11 (self,x):
        return x[0]**4 + x[1]**2
    def f12 (self,x):
        return np.sum((np.array(x)-2)**2)
    def f13 (self,x):
        return 2*np.exp(-x[0]+x[1])
    def f21 (self,x):
        return x[0]**2-2*x[0]+x[1]**2-4*x[1]+4
    def f22 (self,x):
        return 2*x[0]**2-5*x[0]+x[1]**2-2*x[1]+4
    def f23 (self,x):
        return x[0]**2+2*x[1]**2-4*x[1]+1

    def _f1(self, x) -> float:
        return np.max([self.f11(x) ,self.f12(x), self.f13(x)]) + self.f21(x) + self.f22(x) + self.f23(x)
    
    def _f2(self,x ) -> float:
        return max(self.f21( x) + self.f22( x), self.f22( x) + self.f23( x),  self.f21( x) + self.f23( x))
    
    def _gf1(self, x):
        i = np.argmax([self.f11( x),self.f12( x), self.f13( x)])
        if i == 0:
            d = [4*x[0]**3,2*x[1]]
        elif i == 1:
            d = [2*x[0]-4,2*x[1]-4]
        else:
            d = [-2*np.exp(x[1]-x[0]),2*np.exp(x[1]-x[0])]
        return np.add(d,[2*x[0]-2 + 4*x[0] - 5 + 2*x[0], 2*x[1]-4 + 2*x[1]-2 + 4*x[1]-4])

    def _gf2(self, x):
        i = np.argmax([self.f21( x) + self.f22( x), self.f22( x) + self.f23( x),  self.f21( x) + self.f23( x)])
        if i == 0:
            return np.array([2*x[0]-2 + 4*x[0]-5 , 2*x[1]-4 + 2*x[1]-2])
        elif i == 1:
            return np.array([4*x[0]-5 + 2*x[0],2*x[1]-2 + 4*x[1]-4])
        else:
            return np.array([2*x[0]-2 + 2*x[0], 2*x[1]-4 + 4*x[1]-4])

class JB02 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([-1.2,1.])
        self.optimumPoint = np.array([1.,1.])
        self.optimumValue = 0.0

    def _f1(self, x) -> float:
        return abs(x[0]-1) + 200*max([0,abs(x[0])-x[1]])
    
    def _gf1(self, x):
        return np.array([sign(x[0]-1) + 200*(0 if 0 > abs(x[0])-x[1] else sign(x[0])), 200*(0 if 0 > abs(x[0])-x[1] else -1)])

    def _f2(self,x ) -> float:
        return 100*(abs(x[0])-x[1])

    def _gf2(self, x):
        return np.array([100*sign(x[0]), -100])

class JB03 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 4:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([1.,3.,3.,1.])
        self.optimumPoint = np.array([1.,1.,1.,1.])
        self.optimumValue = 0.0

    def _f1(self, x) -> float:
        return abs(x[0]-1) + 200*np.max([0,abs(x[0])-x[1]]) + 180*np.max([0,abs(x[2])-x[3]])+ abs(x[2]-1) +10.1*(abs(x[1]-1)+abs(x[3]-1))+4.95*abs(x[1]+x[3]-2)
    
    def _f2(self,x ) -> float:
        return 100*(abs(x[0])-x[1]) + 90*(abs(x[2])-x[3])+ 4.95*abs(x[1]-x[3])

    def _gf1(self, x): 
        return np.array([sign(x[0]-1) + 200*(0 if 0 > abs(x[0])-x[1] else sign(x[0])), 
                        200*(0 if 0 > abs(x[0])-x[1] else -1) + 10.1*sign(x[1]-1) + 4.95*sign(x[1]+x[3]-2),
                        180*(0 if 0 > abs(x[2])-x[3] else sign(x[2])) +  sign(x[2]-1),
                        180*(0 if 0 > abs(x[2])-x[3] else -1) + 10.1*sign(x[3]-1) + 4.95*sign(x[1]+x[3]-2)
                        ])

    def _gf2(self, x):
        return np.array([100*sign(x[0]), 
                        -100 + 4.95*sign(x[1]-x[3]),
                        90*sign(x[2]),
                        -90 - 4.95*sign(x[1]-x[3])
                        ])

class JB04 (Kernel):
    def __inizialize__(self, dimension):
        self.dimension = dimension # 2 .. 750
        self.XStart = np.array([_ if _ < (dimension+1)/2  else -_ for _ in range(1,dimension+1)])
        self.optimumPoint = np.ones(dimension)
        self.optimumValue = 0.0

    def _f1(self, x):
        return self.dimension * np.max(np.abs(x))
    
    def _f2(self,x ):
        return np.sum(np.abs(x))

    def _gf1(self, x): 
        a = np.zeros (self.dimension)
        i = np.argmax(np.abs(x))
        a[i] = self.dimension*sign(x[i])
        return a

    def _gf2(self, x): 
        return np.array(sign(x)) #broadcast

class JB05 (Kernel):
    def __inizialize__(self, dimension):
        self.XStart = np.zeros(dimension)
        self.XStart[0] = 1./dimension #"bagirov OMS 2002.pdf"
        self.optimumPoint = np.ones(dimension)/self.dimension
        self.optimumValue = 0.0
        self.tm = np.outer(np.arange(1,21)*0.05,np.ones(dimension)) ** np.arange(dimension)

    def ff(self,x):
        return self.tm @ (x - self.optimumPoint)
    
    def _f1(self,x):
        return 20*np.max(abs(self.ff(x)))
    
    def _f2(self,x):
        return np.sum(abs(self.ff(x)))

    def _gf1(self,x):
        r = self.ff(x)
        m = np.argmax(abs(r))
        return 20*(self.tm[m] * np.sign(r[m])) 

    def _gf2(self,x):
        r = self.ff(x)
        return np.sign(r) @ self.tm  

class JB06 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([10.,1.])
        self.optimumPoint = np.array([5.,0.])
        self.optimumValue = -2.5

    def _f1(self,x):
        return x[1] + 0.1*(x[0]**2 + x[1]**2) + 10*max(0,-x[1])

    def _f2(self,x):
        return abs(x[0]) + abs(x[1])

    def _gf1(self,x):
        return np.array([
            .2*x[0] ,
            1 + .2*x[1] + (0 if 0 > -x[1] else -10)
        ])

    def _gf2(self,x):
        return np.array([sign(x[0]), sign(x[1])])

class JB07 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 2:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([-2.,1.])
        self.optimumPoint = np.array([.5,.5])
        self.optimumValue = 0.5

    def f11(self,x):
        return  x[0]**2 + x[1]**2 + abs(x[1])
    def f12(self,x):
        return x[0]+x[0]**2+x[1]**2+abs(x[1]) - .5
    def f13(self,x):
        return abs(x[0]-x[1]) + abs(x[1]) - 1.
    def f14(self,x):
        return  x[0]+x[0]**2+x[1]**2
        
    def _f1(self, x) -> float:
        return abs(x[0]-1) + 200.*np.max([0,abs(x[0])-x[1]]) + 10.*np.max([self.f11(x),self.f12(x),self.f13(x),self.f14(x)])
    
    def _gf1(self, x):
        match np.argmax([self.f11( x),self.f12( x), self.f13( x), self.f14( x)]):
            case 0:
                d = [2.*x[0], 2.*x[1]+sign(x[1])]
            case 1:
                d = [1.+2.*x[0], 2.*x[1]+sign(x[1])]
            case 2:
                d = [sign(x[0]-x[1]), sign(x[0]-x[1])*-1.+sign(x[1])]
            case 3:
                d = [1.+2.*x[0], 2.*x[1]]
            case _: # type 
                d = -1
        d = np.array(d) 

        return np.array([
                    sign(x[0]-1.) + 200.*(0. if 0. > abs(x[0])-x[1] else sign(x[0])),
                        200.*(0. if 0. > abs(x[0])-x[1] else -1.)
                ] + 10*d) 

    def _f2(self,x) -> float:
        return 100.*(abs(x[0])-x[1]) + 10.*(x[0]**2+x[1]**2+abs(x[1]))

    def _gf2(self, x):
        return np.array([
            100.*sign(x[0]) + 20.*x[0], 
            -100. +10.*(2.*x[1]+sign(x[1]))
            ])

class JB08 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 3:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([.5,.5,.5])
        self.optimumPoint = np.array([.75,1.25,.25])
        self.optimumValue = 3.5

    def _f1(self, x) -> float:
        return  9.-8.*x[0]-6*x[1]-4.*x[2]+2.*abs(x[0])+2.*abs(x[1])+2.*abs(x[2])\
            + 4.*x[0]**2 + 2.*x[1]**2 + 2.*x[2]**2.\
            + 10.*np.max([0,x[0]+x[1]+2.*x[2]-3.,-x[0],-x[1],-x[2]])
    
    def _gf1(self, x):
        match np.argmax([0,x[0]+x[1]+2.*x[2]-3.,-x[0],-x[1],-x[2]]):
            case  0:
                d = np.array([0.,0.,0.])
            case  1:
                d = np.array([1.,1.,2.])
            case  2:
                d = np.array([-1.,0.,0.])
            case  3:
                d = np.array([0.,-1.,0.])
            case  4:
                d = np.array([0.,0.,-1.])
            case _ : # type warning 
                d = np.array(0)

        return np.array([
                    -8. + 2.*sign(x[0])+8.*x[0],
                    -6. + 2.*sign(x[1])+4.*x[1],
                    -4. + 2.*sign(x[2])+4.*x[2]
                ] + 10.*d) 

    def _f2(self,x) -> float:
        return abs(x[0]-x[1]) + abs(x[0]-x[2])

    def _gf2(self, x):
        return np.array([
                sign(x[0]- x[1]) + sign(x[0]- x[2]),
                -sign(x[0]- x[1]),
                -sign(x[0]- x[2])
            ])

class JB09 (Kernel):
    def __inizialize__(self, dimension):
        if dimension != 4:
            raise ValueError(f'{self}: dimension {dimension} not supported ')
        self.XStart = np.array([4.,2.,4.,2])
        self.optimumPoint = np.array([7/3.,1./3,.5,2.])
        self.optimumValue = 11/6.

    def _f1(self, x) -> float:
        return    x[0]**2+(x[0]-1)**2+2*(x[0]-2)**2+(x[0]-3)**2\
                + 2*x[1]**2+(x[1]-1)**2+2*(x[1]-2)**2\
                + x[2]**2+(x[2]-1)**2+2*(x[2]-2)**2+(x[2]-3)**2\
                + 2*x[3]**2+(x[3]-1)**2+2*(x[3]-2)**2

    def _gf1(self, x):
        return np.array([
                    2*x[0]+2*(x[0]-1)+4*(x[0]-2)+2*(x[0]-3),
                    4*x[1]+2*(x[1]-1)+4*(x[1]-2),
                    2*x[2]+2*(x[2]-1)+4*(x[2]-2)+2*(x[2]-3),
                    4*x[3]+2*(x[3]-1)+4*(x[3]-2)
                ]) 

    def _f2(self,x) -> float:
        t11 = (x[0]-2)**2+x[1]**2
        t12 = (x[2]-2)**2+x[3]**2
        t21 = (x[0]-2)**2+(x[1]-1)**2
        t22 = (x[2]-2)**2+(x[3]-1)**2
        t31 = (x[0]-3)**2+x[1]**2
        t32 = (x[2]-3)**2+x[3]**2
        t41 = (x[0])**2+(x[1]-2)**2
        t42 = (x[2])**2+(x[3]-2)**2
        t51 = (x[0]-1)**2+(x[1]-2)**2
        t52 = (x[2]-1)**2+(x[3]-2)**2

        return (t11 if t11 > t12 else t12) \
            + (t21 if t21 > t22 else t22) \
            + (t31 if t31 > t32 else t32) \
            + (t41 if t41 > t42 else t42) \
            + (t51 if t51 > t52 else t52) 

    def _gf2(self, x):
        t11 = (x[0]-2)**2+x[1]**2
        t12 = (x[2]-2)**2+x[3]**2
        t21 = (x[0]-2)**2+(x[1]-1)**2
        t22 = (x[2]-2)**2+(x[3]-1)**2
        t31 = (x[0]-3)**2+x[1]**2
        t32 = (x[2]-3)**2+x[3]**2
        t41 = (x[0])**2+(x[1]-2)**2
        t42 = (x[2])**2+(x[3]-2)**2
        t51 = (x[0]-1)**2+(x[1]-2)**2
        t52 = (x[2]-1)**2+(x[3]-2)**2

        return np.array([
                (2*(x[0]-2) if t11 > t12 else 0.)\
                +(2*(x[0]-2) if t21 > t22 else 0.)\
                +(2*(x[0]-.3) if t31 > t32 else 0.)\
                +(2*(x[0]) if t41 > t42 else 0.)\
                +(2*(x[0]-1) if t51 > t52 else 0.),
                
                (2*(x[1]) if t11 > t12 else 0.)\
                +(2*(x[1]-1) if t21 > t22 else 0.)\
                +(2*(x[1]) if t31 > t32 else 0.)\
               +(2*(x[1]-2) if t41 > t42 else 0.)\
                +(2*(x[1]-2) if t51 > t52 else 0.),

                (2*(x[2]-2) if t11 <= t12 else 0.)\
                +(2*(x[2]-2) if t21 <= t22 else 0.)\
                +(2*(x[2]-3) if t31 <= t32 else 0.)\
                +(2*(x[2]) if t41 <= t42 else 0.)\
                +(2*(x[2]-1) if t51 <= t52 else 0.),
                
                (2*(x[3]) if t11 <= t12 else 0.)\
                +(2*(x[3]-1) if t21 <= t22 else 0.)\
                +(2*(x[3]) if t31 <= t32 else 0.)\
                +(2*(x[3]-2) if t41 <= t42 else 0.)\
                +(2*(x[3]-2) if t51 <= t52 else 0.)     
            ])

class JB10 (Kernel):
    def __inizialize__(self, dimension):
        self.XStart = np.array([_*.1 for _ in range (1,dimension+1)])
        self.optimumValue = 2.5 - dimension if dimension%2 > 0 else 1.5 - dimension

        def optPoint(dimension):
            t = np.ones(dimension)
            d = dimension//2
            if dimension%2 > 0:
                for i in range(1,d):
                    if (i%2 == 0):
                        t[i] = -1.
                for i in range(d,dimension):
                    if (i%2 > 0):
                        t[i] = -1.
                t[d] = .0 
            else:
                for i in range(2,dimension,2):
                        t[i] = -1.
            t[0] =-.5
            t[-1] =.5
            return t

        self.optimumPoint = optPoint(dimension) 

    def _f1(self, x):
        return np.sum(x**2)
    
    def _f2(self,x ):
        r = 0.
        for i in range(1,self.dimension):
            r += abs(x[i]-x[i-1])
        return r

    def _gf1(self, x): 
        return np.array(x)*2.

    def _gf2(self, x): 
        l = len(x)
        t = np.empty(l)
        for i in range(len(t)): 
            if i == 0:
                t[0] = -sign(x[1]-x[0])
            elif i == l-1:
                t[-1] = sign(x[-1]-x[-2])
            else:
                t[i] = sign(x[i]-x[i-1]) - sign(x[i+1]-x[i])
        return t

probList_DCJBKM = [ # number are X dimensions
            (JB01,[2]), 
            (JB02,[2]), 
            (JB03,[4]), 
            (JB04,[2,5,10,50,100,150,200,250,500,750
                   ]),
            (JB05,[ 2,5,10,50, 100,150, 200 ,250,300,350,400,500,1000,1500,3000,10000,15000,20000,50000
                   ]),
            (JB06,[2]),
            (JB07,[2]),
            (JB08,[3]),
            (JB09,[4]),
            (JB10,[2,4,5,10,20,50,100,150,200
                   ]),
        ]
