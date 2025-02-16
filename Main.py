# Main.py

import numpy as np
from primal_dual import primal_dual_interior_point
from numerical_utils import scale_problem
from scipy.optimize import linprog
import time
import matplotlib.pyplot as plt


def generate_random_lp(m, n):
    """
    Genera un problema LP factible con A de rango completo y bien escalado.
    """
    # Generar A con rango completo
    while True:
        A = np.random.randn(m, n)
        if np.linalg.matrix_rank(A) == m:
            break
    
    # Generar x > 0 y calcular b = A @ x
    x_true = np.abs(np.random.randn(n)) + 1.0  # x >= 1.0 para evitar valores cercanos a cero
    b = A @ x_true
    
    # Vector de costo aleatorio
    c = np.random.randn(n)
    
    return A, b, c



def generate_large_random_lp(m, n):
    """
    Genera un problema LP factible más grande para probar el algoritmo.
    """
    while True:
        A = np.random.randn(m, n)
        if np.linalg.matrix_rank(A) == m:
            break
    
    x_true = np.abs(np.random.randn(n)) + 1.0
    b = A @ x_true
    c = np.random.randn(n)
    
    return A, b, c


def compare_with_simplex(A, b, c):
    """
    Compara el algoritmo primal-dual con el método Simplex.
    """
    print("\n--- Comparación del Algoritmo Primal-Dual con Simplex ---")

    # Método Simplex
    start_time = time.time()
    res = linprog(c, A_eq=A, b_eq=b, method='highs')
    simplex_time = time.time() - start_time
    print(f"Tiempo de ejecución (Simplex): {simplex_time:.3f} segundos")

    # Verificar si Simplex encontró una solución
    if res.x is None:
        print(f"❌ Simplex no encontró una solución, x = {res.x}. Terminando el proceso.")
        exit()  

    # Algoritmo Primal-Dual
    start_time = time.time()
    x_pd, lam_pd, s_pd, history = primal_dual_interior_point(A, b, c)
    primal_dual_time = time.time() - start_time
    print(f"Tiempo de ejecución (Primal-Dual): {primal_dual_time:.3f} segundos")

    print("\nResultados:")
    print(f"Primal-Dual: x = {x_pd}")
    print(f"Simplex: x = {res.x}")

    # Comparación de tiempos de ejecución
    methods = ['Primal-Dual', 'Simplex']
    times = [primal_dual_time, simplex_time]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, times, color=['blue', 'orange'])
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de Tiempo de Ejecución')
    plt.show()

    # Gráfico de convergencia mejorado
    plt.figure(figsize=(12, 8))

    # Evolución del parámetro de barrera μ en escala logarítmica
    plt.subplot(3, 1, 1)
    plt.plot(history['mu'], label='Parámetro de barrera (μ)', color='purple')
    plt.xlabel('Iteraciones')
    plt.ylabel('μ')
    plt.yscale('log')  # Escala logarítmica para ver la convergencia más clara
    plt.title('Convergencia del Algoritmo Primal-Dual')
    plt.legend()

    # Evolución de los residuos primal y dual
    plt.subplot(3, 1, 2)
    plt.plot(history['residual_primal'], label='Residuo Primal', color='blue')
    plt.plot(history['residual_dual'], label='Residuo Dual', color='red')
    plt.xlabel('Iteraciones')
    plt.ylabel('Norma del Residuo')
    plt.yscale('log')  # Escala logarítmica
    plt.title('Evolución de los Residuos')
    plt.legend()

    # Comparación de soluciones (Primal-Dual vs Simplex)
    plt.subplot(3, 1, 3)
    plt.bar(range(len(x_pd)), x_pd, alpha=0.7, label='Primal-Dual', color='blue')
    plt.bar(range(len(res.x)), res.x, alpha=0.7, label='Simplex', color='orange')
    plt.xlabel('Índice de Variable')
    plt.ylabel('Valor de x')
    plt.title('Comparación de Soluciones Primal-Dual vs Simplex')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Histogramas de la distribución de los valores de x
    plt.figure(figsize=(10, 5))

    plt.hist(x_pd, bins=20, alpha=0.7, label='Primal-Dual', color='blue', edgecolor='black')
    plt.hist(res.x, bins=20, alpha=0.7, label='Simplex', color='orange', edgecolor='black')
    plt.xlabel('Valor de x')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Valores de x (Primal-Dual vs Simplex)')
    plt.legend()

    plt.show()


def generate_test_problem():
    A = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])
    b = np.array([2, 2, 2])
    c = np.array([1, 2, 3])
    return A, b, c


if __name__ == "__main__":
    # Generar un problema aleatorio pequeño para probar
    # A, b, c = generate_random_lp(7, 10)
    # A, b, c = generate_test_problem()
    A, b, c = generate_large_random_lp(50, 100)  # Problema de mayor tamaño (50 restricciones y 100 variables)
    
    A, b, c = scale_problem(A, b, c)  # Escalar el problema antes de resolverlo

    compare_with_simplex(A, b, c)



