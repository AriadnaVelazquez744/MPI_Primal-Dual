# primal_dual.py

import numpy as np
from numerical_utils import solve_newton_system, backtracking_line_search, scale_problem, compute_residuals
from scipy.optimize import linprog
import cvxpy as cp

def find_initial_point_cvxpy(A, b, c):
    import cvxpy as cp
    m, n = A.shape

    # Variables
    x = cp.Variable(n)
    s = cp.Variable(n)

    # Restricciones para asegurar que x > 1e-4 y s > 1e-4
    constraints = [A @ x == b, x >= 1e-4, s >= 1e-4]
    objective = cp.Minimize(cp.sum(s))  # Minimizar suma de holguras

    # Resolver el problema
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        raise ValueError("Fase I (cvxpy) falló. El problema puede ser no factible.")

    # Generar valores iniciales
    lam = np.random.rand(m)
    s = np.maximum(c - A.T @ lam, 1e-4)  # Evitar valores muy pequeños

    return x.value, lam, s


def primal_dual_interior_point(A, b, c, tol=1e-8, max_iter=100, mu_factor=0.125, find_initial_point=None):
    # Escalar el problema para evitar mal condicionamiento
    A, b, c = scale_problem(A, b, c)
    
    # Si no se proporciona un método, usar CVXPY por defecto
    if find_initial_point is None:
        find_initial_point = find_initial_point_cvxpy

    # Obtener el punto inicial
    x, lam, s = find_initial_point(A, b, c)
    m, n = A.shape
    history = {'mu': [], 'residual_primal': [], 'residual_dual': []}
    
    for it in range(max_iter):
        # Calcular el parámetro de centralidad
        mu = np.dot(x, s) / n
        history['mu'].append(mu)
        
        # Calcular los residuos
        r_primal, r_dual, r_cent = compute_residuals(A, x, lam, s, b, c, mu)
        history['residual_primal'].append(np.linalg.norm(r_primal))
        history['residual_dual'].append(np.linalg.norm(r_dual))
        
        # Verificar convergencia
        if mu < tol and np.linalg.norm(r_primal) < tol and np.linalg.norm(r_dual) < tol:
            print(f"Convergencia en iteración {it}: μ = {mu:.3e}")
            break
        
        # Resolver sistema de Newton con la nueva función robusta
        dx, dlam, ds = solve_newton_system(A, x, lam, s, b, c, mu)
        
        # Búsqueda de línea mejorada
        alpha = backtracking_line_search(x, dx, s, ds)
        
        # Actualizar las variables
        x += alpha * dx
        lam += alpha * dlam
        s += alpha * ds
        
        # Reducir el parámetro de centralidad μ
        mu *= mu_factor
        
        # Imprimir progreso
        print(f"Iteración {it}: μ = {mu:.3e}, α = {alpha:.3f}, Residuo primal = {np.linalg.norm(r_primal):.3e}, Residuo dual = {np.linalg.norm(r_dual):.3e}")
    
    else:
        print(f"No convergió en {max_iter} iteraciones. Último μ: {mu:.3e}")
    
    return x, lam, s, history


def find_initial_point_robust(A, b, c):
    """
    Método alternativo para encontrar un punto inicial sin usar CVXPY.
    """
    m, n = A.shape

    # Punto inicial factible
    x0 = np.abs(np.random.rand(n)) + 0.1  # Asegurar x >= 0.1
    lam0 = np.random.rand(m)
    s0 = np.maximum(c - A.T @ lam0, 0.1)  # Asegurar s >= 0.1

    return x0, lam0, s0
