
from algormeter import *
from numpy.linalg import norm
from math import sqrt, log

def gradient(p, **kwargs):
    '''Simple gradient'''
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / norm(p.gfXk) 


df, pv = algorMeter(algorithms = [gradient], problems = probList_covx, iterations = 500, absTol=1E-2)

print('\n', df)
print('\n', pv)
