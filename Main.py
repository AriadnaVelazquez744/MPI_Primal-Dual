# Main.py

import numpy as np
from primal_dual import primal_dual_interior_point, find_initial_point_cvxpy, find_initial_point_robust
from numerical_utils import scale_problem
from scipy.optimize import linprog
import time
import matplotlib.pyplot as plt

from network_flow import generate_network_flow_problem
from transport_problem import generate_transport_problem


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


def compare_with_simplex(A, b, c, use_cvxpy=True):
    """
    Compara el algoritmo primal-dual con el método Simplex.
    """
    print("\n--- Comparación del Algoritmo Primal-Dual con Simplex ---")

    # Elegir el método de punto inicial
    if use_cvxpy:
        print("Usando find_initial_point_cvxpy...")
        find_initial_point = find_initial_point_cvxpy
    else:
        print("Usando find_initial_point_robust...")
        find_initial_point = find_initial_point_robust

    # Algoritmo Primal-Dual
    start_time = time.time()
    x_pd, lam_pd, s_pd, history = primal_dual_interior_point(A, b, c, find_initial_point=find_initial_point)
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

    # Gráfico de comparación del tiempo
    methods = ['Primal-Dual', 'Simplex']
    times = [primal_dual_time, simplex_time]
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods, times, color=['blue', 'orange'])
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de tiempo de ejecución')
    plt.show()

    print(f"Tiempo de ejecución (Primal-Dual): {primal_dual_time:.3f} segundos")
    print(f"Tiempo de ejecución (Simplex): {simplex_time:.3f} segundos")

    # Visualización de la convergencia del parámetro μ
    plt.figure(figsize=(10, 6))
    plt.plot(history['mu'], label='Primal-Dual: μ vs Iteraciones')
    plt.xlabel('Iteraciones')
    plt.ylabel('Parámetro de barrera (μ)')
    plt.title('Convergencia del Algoritmo Primal-Dual')
    plt.legend()
    plt.show()



    # Gráfico descriptivo
    plt.figure(figsize=(12, 8))

    # Gráfico del parámetro μ
    plt.subplot(2, 1, 1)
    plt.plot(history['mu'], label='Parámetro de barrera (μ)')
    plt.xlabel('Iteraciones')
    plt.ylabel('μ')
    plt.title('Convergencia del Algoritmo Primal-Dual')
    plt.legend()
    
    # Gráfico del residuo primal y dual
    plt.subplot(2, 1, 2)
    plt.plot(history['residual_primal'], label='Residuo Primal')
    plt.plot(history['residual_dual'], label='Residuo Dual')
    plt.xlabel('Iteraciones')
    plt.ylabel('Norma del Residuo')
    plt.title('Evolución de los Residuos')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
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
    # A, b, c = generate_large_random_lp(50, 100)  # Problema de mayor tamaño (50 restricciones y 100 variables)
    
    # Cambiar el tipo de problema según lo que quieras probar
    problem_type = "transport"  # Cambia a "network" para el problema de flujo de red
    # problem_type = "network"  # Cambia a "transport" para el problema de transporte

    if problem_type == "network":
        A, b, c = generate_network_flow_problem()
    elif problem_type == "transport":
        A, b, c = generate_transport_problem()
    else:
        raise ValueError("Tipo de problema no válido. Elige 'network' o 'transport'.")

    A, b, c = scale_problem(A, b, c)  # Escalar el problema antes de resolverlo

    compare_with_simplex(A, b, c, use_cvxpy=True)



