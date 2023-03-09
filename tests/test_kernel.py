from algormeter.tools import counter, dbx
from algormeter.libs import *

def test_cache():
    p = Parab(2)
    x = np.ones(2)
    p.f1(x)
    assert counter.get('f1') == 1
    p.f1(x)
    assert counter.get('f1') == 1
    assert counter.get('f2') == None 
    p.f(x)
    assert counter.get('f1') == 1
    assert counter.get('f2') == 1 
    assert counter.get('gf2') == None 
    p.gf(x)
    assert counter.get('f1') == 1
    assert counter.get('f2') == 1 
    assert counter.get('gf2') == 1 

def test_skipcache():
    p = Parab(2)
    x = np.ones(2)
    p._f1(x)
    assert counter.get('f1') is None 

def test_cacheSize():
    p = Parab(2)
    for i in range(Kernel.CACHESIZE):
        x = np.zeros(2)+i
        p.f(x)
    assert counter.get('f1') == Kernel.CACHESIZE 
    p.f(x) # type: ignore
    assert counter.get('f1') == Kernel.CACHESIZE 

def test_cache_life(): 
    p = Parab(2)
    y = -np.ones(2)
    for i in range(Kernel.CACHESIZE-1):
        p.f(y)
        x = np.zeros(2)+i
        p.f(x)
    assert counter.get('f1') == Kernel.CACHESIZE 
    p.f(y)
    assert counter.get('f1') == Kernel.CACHESIZE 


def test_explib():
    def explib(lib):
        for el, dims in lib:
            for d in dims:
                p = el(d)
                assert np.isclose(p.f(p.optimumPoint), p.optimumValue) 
    explib(probList_DCJBKM)
    explib(probList_covx)
    explib(probList_base)

def test_isMinimum():
    p = JB01(2)
    assert p.isMinimum(np.array([1,1])) == True
    assert p.isMinimum(np.array([1,-2])) == False
    p = JB04(2)
    assert p.isMinimum(np.array([1,-1])) == True
    assert p.isMinimum(np.array([1,-2])) == False
