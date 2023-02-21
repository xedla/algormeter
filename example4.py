# tuneParams example
import pandas as pd

from algormeter import *
import algormeter.algorithms as alg

Param.alpha = 1. # type: ignore
Param.beta = 1. # type: ignore

def gradient(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - Param.alpha/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) # type: ignore

tpar = [ # [name, [values list]]
    ('Param.alpha', [1. + i for i in np.arange(.05,.9,.05)]),
    # ('Param.beta', [1., 2., 3.]),
]

problems = [
            (Parab,[2,4,6]),
            (ParAbs,[2]),
            (Acad,[2]),

            # (JB01,[2]), 
            # (JB02,[2]), 
            # (JB03,[4]),
            # (JB04,[2,5,10]),
            # (JB05,[2,5,10]),
            # (JB06,[2]),
            # (JB07,[2]), 
            # (JB08,[3]),
            # (JB09,[4]),
            # (JB10,[2,4,5]),
        ]

algorithms = [
                # alg.polyak,
                # alg.loggradient,
                # alg.gradient,
                gradient,
        ]

dff, pv = algorMeter(algorithms = algorithms, problems = problems, iterations = 3000,
                    tuneParameters=tpar, 
                    #  trace=True, 
                     # dbprint = True 
                     )

print('\n', dff)
print('\n', pv)



df = dff[dff.Status == 'Success']
# df = df[ df.Param.str.contains("alpha': 0.3") ]

df = df.groupby(['Param','Status',]).agg({'Status':'count','f1':'sum'})
df.rename(columns={'Status': 'count'}, inplace=True)
df = df.sort_values(['count','f1'],ascending=[False, True])

pd.options.display.max_rows = 2000
pd.options.display.max_colwidth = 1000

print(df)

# or better to use spreadsheet pivot from csv
