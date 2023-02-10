# minimize method example

from algormeter import *

# define my problem parameters dependent 
class MyProb (Kernel):
    def __inizialize__(self, dimension):
        self.XStart = np.ones(self.dimension)*K #  K param
    def _f1(self, x):
        return np.sum(np.array(x)**2) 
    def _gf1(self, x):
        return 2.*x
    def _f2(self,x) :
        return H #  H param
    def _gf2(self, x):
        return np.zeros(self.dimension)

# ...define my algorithm
def myAlgo(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) 

# ...apply my algorithm to my problem
for H in [3,5]:
    for K in [1,2,4]:
        p = MyProb() 
        found, x, y = p.minimize(myAlgo)

        print(f'With K:{K}, H:{H} found:{found}, y is {y} at {x}')

