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



algoList_simple = [
    gradient,
    harmonicGradient,
    logGradient,
    sqrtGradient,
    polyak
]