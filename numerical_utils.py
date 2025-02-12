# #numerical_utils.py

import numpy as np
from scipy.linalg import lu_factor, lu_solve

# def solve_newton_system(A, x, lam, s, b, c, mu):
#     """
#     Resuelve el sistema KKT regularizado usando descomposición LU.
#     """
#     m, n = A.shape
#     X = np.diag(x)
#     S = np.diag(s)

#     # Matriz KKT regularizada
#     KKT_top = np.hstack([np.zeros((n, n)), A.T, np.eye(n)])
#     KKT_mid = np.hstack([A, np.zeros((m, m + n))])
#     KKT_bot = np.hstack([S, np.zeros((n, m)), X + 1e-6 * np.eye(n)])  # Regularización más fuerte

#     KKT = np.vstack([KKT_top, KKT_mid, KKT_bot])

#     # Vector de residuos
#     r_dual = A.T @ lam + s - c
#     r_primal = A @ x - b
#     r_cent = x * s - mu  # Producto complementario

#     rhs = -np.concatenate([r_dual, r_primal, r_cent])

#     # Resolver sistema con descomposición LU
#     lu, piv = lu_factor(KKT)
#     delta = lu_solve((lu, piv), rhs)

#     dx = delta[:n]
#     dlam = delta[n:n+m]
#     ds = delta[n+m:]

#     return dx, dlam, ds

def solve_newton_system(A, x, lam, s, b, c, mu):
    """
    Resuelve el sistema KKT con mejor regularización y comprobación de singularidad.
    """
    m, n = A.shape
    X = np.diag(x)
    S = np.diag(s)
    
    # Regularización más fuerte para evitar singularidad
    epsilon = 1e-6
    KKT_top = np.hstack([np.zeros((n, n)), A.T, np.eye(n)])
    KKT_mid = np.hstack([A, np.zeros((m, m + n))])
    KKT_bot = np.hstack([S, np.zeros((n, m)), X + epsilon * np.eye(n)])  # Regularización fuerte en X
    
    KKT = np.vstack([KKT_top, KKT_mid, KKT_bot])
    
    # Verificación de valores NaN o Inf antes de resolver el sistema
    if np.any(np.isnan(KKT)) or np.any(np.isinf(KKT)):
        raise ValueError("La matriz KKT contiene NaN o Inf. Verifica el punto inicial o las restricciones.")
    
    # Resolver el sistema con LU factorization
    try:
        lu, piv = lu_factor(KKT)
        delta = lu_solve((lu, piv), -np.concatenate([
            A.T @ lam + s - c,
            A @ x - b,
            x * s - mu  # Producto complementario
        ]))
    except np.linalg.LinAlgError:
        raise ValueError("El sistema KKT es singular. Considera escalar el problema o mejorar el punto inicial.")
    
    dx = delta[:n]
    dlam = delta[n:n+m]
    ds = delta[n+m:]
    
    return dx, dlam, ds

def backtracking_line_search(x, dx, s, ds, alpha=1.0, beta=0.7, min_alpha=1e-10):
    """
    Búsqueda de línea robusta con verificación de factibilidad y reducción del residuo.
    """
    while alpha > min_alpha:
        x_new = x + alpha * dx
        s_new = s + alpha * ds
        if np.all(x_new > 0) and np.all(s_new > 0):
            return alpha
        alpha *= beta
    raise ValueError("Búsqueda de línea fallida. Verifique el punto inicial.")

def scale_problem(A, b, c):
    """
    Escala A, b y c para evitar problemas de mal condicionamiento.
    """
    A_max = np.max(np.abs(A), axis=0)
    A_scaled = A / A_max
    b_scaled = b / np.linalg.norm(b)
    c_scaled = c / np.linalg.norm(c)
    
    return A_scaled, b_scaled, c_scaled
