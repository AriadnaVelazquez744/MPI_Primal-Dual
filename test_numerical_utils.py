import numpy as np
from numerical_utils import solve_newton_system, backtracking_line_search

def test_solve_newton_system():
    A = np.array([[1, 1], [1, -1]])
    b = np.array([1, 0])
    c = np.array([1, 1])
    x = np.array([1, 1])
    lam = np.array([0.5, 0.5])
    s = np.array([1, 1])
    mu = 0.1
    
    dx, dlam, ds = solve_newton_system(A, x, lam, s, b, c, mu)
    print("dx =", dx)
    print("dlam =", dlam)
    print("ds =", ds)

def test_backtracking_line_search():
    x = np.array([1, 1])
    dx = np.array([-0.1, -0.1])
    s = np.array([1, 1])
    ds = np.array([-0.1, -0.1])
    
    alpha = backtracking_line_search(x, dx, s, ds)
    print("alpha =", alpha)

if __name__ == "__main__":
    test_solve_newton_system()
    test_backtracking_line_search()
