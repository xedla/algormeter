# multiple run of each problem with random start point

from algormeter import *
import pandas as pd

def polyak(p, **kwargs):

    p.randomSet(center=10,size=5) 
    
    for k in p.loop():
        if p.gfXk.any():
            p.Xkp1 = p.Xk - (p.fXk - p.optimumValue) * p.gfXk / (np.linalg.norm(p.gfXk)**2) 
        else:
            p.Xkp1 = p.Xk


df, pv = algorMeter(algorithms = [polyak], problems = probList_covx, runs=50, iterations = 50, absTol=1E-2
                    #  trace=True, 
                     # dbprint = True 
                     )

print('\n', df)
print('\n', pv)

df2 = df.groupby(['Algorithm','Problem','Dim','Status']).agg({'Status': 'count','Iterations':['min', 'max','mean'],'Delta':['min', 'max','mean']})
print('\n', df2)
