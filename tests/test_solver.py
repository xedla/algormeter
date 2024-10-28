# minimize method example
from numpy import array, dot, allclose
from qpsolvers import solve_qp
import warnings


M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))
A = array([1., 1., 1.])
b = array([1.])

def test_solver():
    warnings.filterwarnings("ignore") 

    x = solve_qp(P, q, G, h, A, b, solver="scs")
    # print(f"QP solution: x = {x}")
    assert allclose(array(x), array([ 0.30768706, -0.69245387,  1.38476663]))