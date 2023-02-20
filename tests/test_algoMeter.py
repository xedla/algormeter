from algormeter import *
from algormeter.libs import *
import algormeter.algorithms as alg

def test_overall():
    df, pv = algorMeter(algorithms = [alg.gradient], problems = [ (Parab,[2]) ], csv=False, iterations=300)
    print(df.values.tolist())
    # assert df.loc[0,'f1'] == 141
    assert df.loc[0,'Status'] == 'Success'
