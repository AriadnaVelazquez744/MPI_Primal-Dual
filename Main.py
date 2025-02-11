import numpy as np
from primal_dual import primal_dual_interior_point
from scipy.optimize import linprog
import time
import matplotlib.pyplot as plt

def generate_random_lp(m, n):
    """
    Genera un problema LP aleatorio de tamaño (m, n).
    """
    A = np.random.rand(m, n) * 10 - 5
    b = np.random.rand(m) * 10
    c = np.random.rand(n) * 10 - 5
    return A, b, c

def compare_with_simplex(A, b, c):
    """
    Compara el algoritmo primal-dual con el método Simplex.
    """
    print("\n--- Comparación del Algoritmo Primal-Dual con Simplex ---")

    # Algoritmo Primal-Dual
    start_time = time.time()
    x_pd, lam_pd, s_pd, history = primal_dual_interior_point(A, b, c)
    primal_dual_time = time.time() - start_time
    print(f"Tiempo de ejecución (Primal-Dual): {primal_dual_time:.3f} segundos")

    # Método Simplex
    start_time = time.time()
    res = linprog(c, A_eq=A, b_eq=b, method='simplex')
    simplex_time = time.time() - start_time
    print(f"Tiempo de ejecución (Simplex): {simplex_time:.3f} segundos")

    print("\nResultados:")
    print(f"Primal-Dual: x = {x_pd}")
    print(f"Simplex: x = {res.x}")

    # Visualización de la convergencia del parámetro μ
    plt.figure(figsize=(10, 6))
    plt.plot(history['mu'], label='Primal-Dual: μ vs Iteraciones')
    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro de barrera (μ)')
    plt.title('Convergencia del Algoritmo Primal-Dual')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generar un problema aleatorio pequeño para probar
    A, b, c = generate_random_lp(5, 10)
    compare_with_simplex(A, b, c)
