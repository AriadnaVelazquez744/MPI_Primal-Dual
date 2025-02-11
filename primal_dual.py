import numpy as np
from numerical_utils import solve_newton_system, backtracking_line_search

def find_initial_point(A, b, c):
    """
    Encuentra un punto inicial factible para el algoritmo.
    """
    n = A.shape[1]
    x0 = np.ones(n)       # Se asume un punto inicial factible simple
    s0 = np.ones(n)       # Valores positivos para s
    lam0 = np.zeros(A.shape[0])  # Inicialmente cero para los multiplicadores
    return x0, lam0, s0

def primal_dual_interior_point(A, b, c, tol=1e-8, max_iter=100, mu_factor=0.1):
    """
    Implementa el algoritmo primal-dual de puntos interiores.
    """
    x, lam, s = find_initial_point(A, b, c)
    n = len(x)
    
    history = {'mu': [], 'residual_norm': []}
    
    for it in range(max_iter):
        mu = np.dot(x, s) / n
        history['mu'].append(mu)
        
        if mu < tol:
            print(f"Convergencia alcanzada en la iteración {it} con μ = {mu:.3e}")
            break
        
        dx, dlam, ds = solve_newton_system(A, x, lam, s, b, c, mu)
        alpha = backtracking_line_search(x, dx, s, ds)
        
        x = x + alpha * dx
        lam = lam + alpha * dlam
        s = s + alpha * ds
        
        residual_norm = np.linalg.norm(A.dot(x) - b) + np.linalg.norm(A.T.dot(lam) + s - c)
        history['residual_norm'].append(residual_norm)
        
        print(f"Iteración {it}: μ = {mu:.3e}, α = {alpha:.3f}, residual_norm = {residual_norm:.3e}")
    
    return x, lam, s, history
