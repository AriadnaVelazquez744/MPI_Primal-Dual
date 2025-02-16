# problems.py
import numpy as np

# Problema 1 (Ejercicio 1)
def problema_1():
    A = np.array([
        [1, 1, 1, 0, 0],
        [1, 2, 0, 1, 0],
        [-1, 1, 0, 0, 1]
    ])
    b = np.array([3, 4, 1])
    c = np.array([-2, -3, 0, 0, 0])  # Maximizar → min -c^T x
    return A, b, c

# Problema 4a (f(x1, x2) = x1)
def problema_4a():
    A = np.array([
        [1, -1, 1, 0],
        [-4, 1, 0, 1]
    ])
    b = np.array([1, 4])
    c = np.array([1, 1, 0, 0])
    return A, b, c

# Problema 4b (f(x1, x2) = -2x1 + x2)
def problema_4b():
    A = np.array([
        [1, 1, 1, 0],   # x1 - x2 + s1 = 1
        [-4, 1, 0, 1]    # -4x1 + x2 + s2 = 4
    ])
    b = np.array([1, 4])
    c = np.array([-2, 1, 0, 0])  # min -2x1 + x2
    return A, b, c
# Problema 4c (f(x1, x2) = 8x1 - 2x2)
def problema_4c():
    A = np.array([
        [1, -1, 1, 0],
        [-4, 1, 0, 1]
    ])
    b = np.array([1, 4])
    c = np.array([8, -2, 0, 0])
    return A, b, c

# Problema 5 (Ejercicio 5)
# Problema 5 (solución directa sin variables artificiales)
def problema_5():
    A = np.array([
        [1, -1, 1],    # x1 - x2 - x3 = 1
        [2, -3, 3]     # 2x1 -3x2 -3x3 = 2
    ])
    b = np.array([1, 2])
    c = np.array([1, 1, 4])  # min x1 + x2 + 4x3
    return A, b, c

# Ejecución directa desde problems.py
if __name__ == "__main__":
    from Main import compare_with_simplex, scale_problem

    problemas = {
        "Ejercicio 1": problema_1(),
        "Ejercicio 4a": problema_4a(),
        "Ejercicio 4b": problema_4b(),
        "Ejercicio 4c": problema_4c(),
        "Ejercicio 5": problema_5()
    }

    for nombre, (A, b, c) in problemas.items():
        print(f"\n=== Resolviendo {nombre} ===")
        A_scaled, b_scaled, c_scaled = scale_problem(A, b, c)
        compare_with_simplex(A_scaled, b_scaled, c_scaled)