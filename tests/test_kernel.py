from algormeter.libs import *

def test_cache():
    p = Parab(2)
    p.f1([1,1])
    assert counter.get('f1') == 1
    p.f1([1,1])
    assert counter.get('f1') == 1
    assert counter.get('f2') == None 
    p.f([1,1])
    assert counter.get('f1') == 1
    assert counter.get('f2') == 1 
    assert counter.get('gf2') == None 
    p.gf([1,1])
    assert counter.get('f1') == 1
    assert counter.get('f2') == 1 
    assert counter.get('gf2') == 1 

def test_cacheSize():
    p = Parab(2)
    for i in range(Kernel.CACHESIZE):
        x = np.zeros(2)+i
        p.f(x)
    assert counter.get('f1') == Kernel.CACHESIZE 
    p.f(x)
    assert counter.get('f1') == Kernel.CACHESIZE 


def test_explib():
    def explib(lib):
        for el, dims in lib:
            for d in dims:
                p = el(d)
                assert np.isclose(p.f(p.optimumPoint), p.optimumValue) 
    explib(probList_DCJBKM)
    explib(probList_coax)
    explib(probList_base)