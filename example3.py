# dbx.print  tuneParam example

from algormeter import *
from numpy.linalg import norm
from math import sqrt, log

TunePar.TMAX = 3. # type: ignore
TunePar.TMIN = .05 # type: ignore
TunePar.EPS = .01 # type: ignore
tpar = [ # [name, [values list]]
    ('TunePar.TMAX', [i for i in np.arange(1.,10.,1.)]),
    ('TunePar.TMIN', [i for i in np.arange(.05,1.,.05)]),
]


def NSBB(p, **kwargs):
    def t():
        d = p.Xk - Xprev
        t = norm(p.gfXk)*norm(d)**2/(TunePar.EPS + 2*(p.f(Xprev)- p.fXk + p.gfXk @ d)) # type: ignore
        dbx.print('t:',t, 'Xprev:',Xprev, 'f(Xprev):',p.f(Xprev) )
        m = 1./sqrt(p.K+1)
        t = max(t, TunePar.TMIN*m) # type: ignore
        t = min(t, TunePar.TMAX*m) # type: ignore
        return t

    def halt():
        return np.isclose(p.fXk,p.optimumValue,atol=1.E-6) 

    p.isHalt = halt
    Xprev = p.XStart + .1
    
    for k in p.loop():
        p.Xkp1 = p.Xk - t() * p.gfXk / norm(p.gfXk) 
        Xprev = p.Xk

df, pv = algorMeter(algorithms = [NSBB], problems = probList_coax, iterations = 2000,
# df, pv = algorMeter(algorithms = [NSBB], problems = [(MAXQ,[20])], iterations = 100, absTol=1E-4
                    # tuneParameters=tpar, 
                    # trace=True,         
                    # dbprint = True 
                     )

print('\n', df)
print('\n', pv)
