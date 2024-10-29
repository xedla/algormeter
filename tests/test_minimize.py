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
    result, x, y = p.minimize(myAlgo)
    print(result)
    assert result == 'Success' and np.isclose(y,-1.), 'minimize failed'

def test_minimize_max_iter():
    p = MyProb(K)
    result, x, y = p.minimize(myAlgo, iterations=10)
    assert result == 'MaxIter' , 'minimize must fail for max iterations'
