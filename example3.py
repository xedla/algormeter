# dbx.print trace counter.up, override stop, break

from algormeter import *
from numpy.linalg import norm
from math import sqrt
from algormeter.tools import counter, dbx

TMAX = 3. 
TMIN = .05 
EPS = .01 

# Nonsmooth Barzilai-Borwein (NSBB) algorithm
def NSBB(p, **kwargs):
    def t():
        d = p.Xk - Xprev
        t = norm(p.gfXk)*norm(d)**2/(EPS + 2*(p.f(Xprev)- p.fXk + p.gfXk @ d)) 
        dbx.print('t:',t, 'Xprev:',Xprev, 'f(Xprev):',p.f(Xprev) )
        m = 1./sqrt(p.K+1)
        if t < TMIN*m: 
            t = TMIN*m
            counter.up('min',cls='t')
        if t > TMAX*m: 
            t = TMAX*m
            counter.up('max',cls='t')
        return t

    def halt():
        return np.isclose(p.fXk,p.optimumValue,atol=1.E-6) 
    p.stop = halt

    Xprev = p.XStart + .1
    counter.log('hi', 'msg',cls='Welcome')

    for k in p.loop():
        # if np.isclose(p.fXk,p.optimumValue,atol=1.E-6): # alternative at stop redefine used above  
        #     break

        p.Xkp1 = p.Xk - t() * p.gfXk / norm(p.gfXk) 
        Xprev = p.Xk

df, pv = algorMeter(algorithms = [NSBB], problems = probList_base, iterations = 100,
                    trace=True,         
                    dbprint = True 
                     )

print('\n', df)
print('\n', pv)
