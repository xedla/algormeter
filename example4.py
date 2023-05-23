# tuneParams: algorithm parameters tuning
import pandas as pd

from algormeter import *
import algormeter.algorithms as alg

alpha = 1. 
beta = 1. 

def gradient(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - alpha/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) # type: ignore
    # print (alpha,beta)


tpar : TuneParameters = [ # [name, [values list]]
    ('alpha', [1. + i for i in np.arange(.05,.9,.05)]),
    ('beta', [1., 2., 3.]),
]

problems : Problems = [
            (Parab,[2,4,6]),
            (ParAbs,[2]),
            (Acad,[2]),
        ]

algorithms : Algorithms = [
                gradient,
        ]

dff, pv = algorMeter(algorithms = algorithms, problems = problems, iterations = 3000,
                    tuneParameters=tpar, 
                     )

print('\n', dff)
print('\n', pv)



df = dff[dff.Status == 'Success']
# df = df[ df.TuneParams.str.contains("alpha': 0.3") ]

df = df.groupby(['TuneParams','Status',]).agg({'Status':'count','f1':'sum'})
df.rename(columns={'Status': 'count'}, inplace=True)
df = df.sort_values(['count','f1'],ascending=[False, True])

pd.options.display.max_rows = 2000
pd.options.display.max_colwidth = 1000

print(df)

# or better to use spreadsheet pivot from csv
