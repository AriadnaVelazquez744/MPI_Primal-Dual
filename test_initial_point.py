import numpy as np
from primal_dual import find_initial_point_cvxpy, find_initial_point_robust

def generate_test_problem():
    """
    Genera un problema LP pequeño para probar los métodos de generación de punto inicial.
    """
    A = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])
    b = np.array([2, 2, 2])
    c = np.array([1, 2, 3])
    return A, b, c

def test_find_initial_point():
    A, b, c = generate_test_problem()

    # Probar el método con cvxpy
    print("Probando find_initial_point_cvxpy...")
    try:
        x, lam, s = find_initial_point_cvxpy(A, b, c)
        print("Resultado con cvxpy:")
        print("x =", x)
        print("λ =", lam)
        print("s =", s)
    except Exception as e:
        print("Error en find_initial_point_cvxpy:", e)

    # Probar el método sin cvxpy
    print("\nProbando find_initial_point_robust...")
    try:
        x, lam, s = find_initial_point_robust(A, b, c)
        print("Resultado sin cvxpy:")
        print("x =", x)
        print("λ =", lam)
        print("s =", s)
    except Exception as e:
        print("Error en find_initial_point_robust:", e)

if __name__ == "__main__":
    test_find_initial_point()
