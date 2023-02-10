# minimize method example

from algormeter import *
K = 2

class MyProb (Kernel):
    def __inizialize__(self, dimension):
        self.XStart = np.ones(self.dimension)*K
    def _f1(self, x):
        return np.sum(np.array(x)**2) 
    def _gf1(self, x):
        return 2.*x
    def _f2(self,x) :
        return 1.
    def _gf2(self, x):
        return np.zeros(self.dimension)


def myAlgo(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) 

def test_minimize():
    p = MyProb(K)
    found, x, y = p.minimize(myAlgo)
    assert found == True and np.isclose(y,-1.), 'minimize failed'
