
from algormeter import *
from algormeter.algorithms import *

algorithms = [
                harmonicGradient,
                sqrtGradient,
                logGradient,
                polyak
        ]


df, pv = algorMeter(algorithms = algorithms, problems = probList_covx, iterations = 1000, absTol=1E-2)

print('\n', df)
print('\n', pv)