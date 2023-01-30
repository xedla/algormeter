# tuneParams example

from algormeter import *
import algormeter.algorithms as alg

Param.alpha = 1. # type: ignore

def gradient(p, **kwargs):
    for k in p.loop():
        p.Xkp1 = p.Xk - Param.alpha/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) # type: ignore

tpar = [ # [name, [values list]]
    ('Param.alpha', [1. + i for i in np.arange(.05,.9,.05)]),
    # ('Param.beta', [1. + i for i in np.arange(.05,.9,.05)]),
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
                alg.polyak,
                # alg.loggradient,
                # alg.gradient,
                gradient,
        ]

df, pv = algorMeter(algorithms = algorithms, problems = problems, iterations = 3000,
                    tuneParameters=tpar, 
                    #  trace=True, 
                     # dbprint = True 
                     )

print('\n', df)
print('\n', pv)
