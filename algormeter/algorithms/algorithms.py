from copy import deepcopy
import numpy as np
import math
from algormeter.kernel import *

def gradient(p, **kwargs):
    '''Simple gradient
    '''
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) 

def harmonicGradient(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) 

def logGradient(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(math.log(k+2)) * p.gfXk / np.linalg.norm(p.gfXk) 

def sqrtGradient(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(math.sqrt(k+1)) * p.gfXk / np.linalg.norm(p.gfXk) 

def polyak(p, **kwargs):
    for k in p.loop():
        if p.gfXk.any():
            p.Xkp1 = p.Xk - (p.fXk - p.optimumValue) * p.gfXk / (np.linalg.norm(p.gfXk)**2) 
        else:
            p.Xkp1 = p.Xk

def DCA(p, **kwargs):
    '''DCA difference of two convex functions standard algorithm
    '''

    def phi(x):
        # print(f'f2xk:{f2xk},g2xk:{g2xk}')
        return f2xk + g2xk.T @ (x - xk)

    def gphi(x):
        return g2xk

    q = deepcopy(p)
    q.f2 = phi # sostituisce f2 originale con la sua linearizzazione
    q.gf2 = gphi # e il gradiente
    q.config(trace=False,csv=False,savedata=False,maxiterations=50)

    for k in p.loop():

        f2xk = p.f2Xk
        g2xk = p.gf2Xk 
        xk = p.Xk
        q.setStartPoint(xk)
        (found, x, y) = q.minimize(gradient)
        # print('phi:', found, x, y)
        p.Xkp1 = x


def gradientV1(p, altFunc, evalEveryNumStep, **kwargs):
    ''' gradient variant 1 called by DCAv1
    '''
    if not callable(altFunc): 
        raise ValueError('altFunc is not callable')

    def stop():
        if p.K % evalEveryNumStep == 0:
            return altFunc(p.Xkp1) <  altFunc(p.Xk) - .1 * np.linalg.norm(p.gf1Xk - p.gf2Xk)**2
        return False

    p.stop = stop # sostituisce stop con il nuovo criterio

    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) 


def DCAv1(p, **kwargs):
    '''DCA difference of two convex functions standard algorithm
    '''
    def phi(x):
        # print(f'f2xk:{f2xk},g2xk:{g2xk}')
        return f2xk + g2xk.T @ (x - xk)

    def gphi(x):
        return g2xk

    q = deepcopy(p)
    q.f2 = phi # sostituisce f2 originale con la sua linearizzazione
    q.gf2 = gphi # e il gradiente
    q.config(trace=False,csv=False,savedata=False,maxiterations=50)

    for k in p.loop():

        f2xk = p.f2Xk
        g2xk = p.gf2Xk 
        xk = p.Xk
        q.setStartPoint(xk)
        (found, x, y) = q.minimize(gradientV1, altFunc = p.f, evalEveryNumStep = 10)
        # print('phi:', found, x, y)
        p.Xkp1 = x

algoList_simple = [
    gradient,
    harmonicGradient,
    logGradient,
    sqrtGradient,
    polyak
]