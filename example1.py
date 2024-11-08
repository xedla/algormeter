
from algormeter import *
from numpy.linalg import norm
from math import sqrt, log

def gradient(p, **kwargs):
    '''Simple gradient'''
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / norm(p.gfXk) 


def algpt(p, **kwargs):
    learning_rate=0.01
    tolerance=1e-6
    for k in p.loop():
        p.Xkp1 = p.Xk - learning_rate * p.gfXk
        if np.linalg.norm(p.Xkp1 - p.gfXk) < tolerance:
            break


df, pv = algorMeter(algorithms = [gradient,algpt], problems = probList_covx, iterations = 500, absTol=1E-2)

print('\n', df)
print('\n', pv)
