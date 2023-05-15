''' Dolan, More  performance profile
        [Elizabeth D. Dolan · Jorge J. More ́ Benchmarking Optimization Software with Performance Profiles]]
'''
import pandas as pd
import matplotlib.pyplot as plt
from algormeter import *
from algormeter.algorithms import *

algorithms :Algorithms = [
                harmonicGradient,
                sqrtGradient,
                logGradient,
                polyak
        ]


df, pv = algorMeter(algorithms = algorithms, problems = probList_base + probList_covx, iterations = 2000)

print('\n', df)
print('\n', pv)

# df.to_pickle('perfprof.pickle')
# df = pd.read_pickle('perfprof.pickle')

perfProf(df, costs= ['Iterations','Seconds'] )

plt.show(block=True)
