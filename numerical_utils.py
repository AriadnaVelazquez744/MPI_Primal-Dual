import numpy as np
from scipy.linalg import lu_factor, lu_solve

def solve_newton_system(A, x, lam, s, b, c, mu):
    m, n = A.shape
    X = np.diag(x)
    S = np.diag(s)
    
    # Construir matriz KKT con regularización
    KKT_top = np.hstack([np.zeros((n, n)), A.T, np.eye(n)])
    KKT_mid = np.hstack([A, np.zeros((m, m + n))])
    KKT_bot = np.hstack([S, np.zeros((n, m)), X + 1e-6 * np.eye(n)])  # Aumentar la regularización
    
    KKT = np.vstack([KKT_top, KKT_mid, KKT_bot])
    
    # Vector de residuos
    r_dual = A.T @ lam + s - c
    r_primal = A @ x - b
    r_cent = x * s - mu  # Element-wise
    
    rhs = -np.concatenate([r_dual, r_primal, r_cent])
    
    # Resolver sistema con descomposición LU
    lu, piv = lu_factor(KKT)
    delta = lu_solve((lu, piv), rhs)
    
    dx = delta[:n]
    dlam = delta[n:n+m]
    ds = delta[n+m:]
    
    return dx, dlam, ds

def backtracking_line_search(x, dx, s, ds, alpha=1.0, beta=0.7, min_alpha=1e-10):
    """
    Búsqueda de línea con límite inferior realista.
    """
    while alpha > min_alpha:
        x_new = x + alpha * dx
        s_new = s + alpha * ds
        if np.all(x_new > 0) and np.all(s_new > 0):
            return alpha
        alpha *= beta
    raise ValueError("Búsqueda de línea fallida. Verifique el punto inicial.")