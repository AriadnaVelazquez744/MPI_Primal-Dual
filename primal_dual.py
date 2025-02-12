import numpy as np
from numerical_utils import solve_newton_system, backtracking_line_search

def find_initial_point(A, b, c):
    m, n = A.shape
    
    # Problema de Fase I: min 1*t sujeto a A x + t = b, x >= 0, t >= 0
    A_aug = np.hstack([A, np.ones((m, 1))])
    c_aug = np.zeros(n + 1)
    c_aug[-1] = 1  # Minimizar t
    
    # Resolver con linprog usando HiGHS para evitar warnings
    from scipy.optimize import linprog
    res = linprog(c_aug, A_eq=A_aug, b_eq=b, bounds=[(0, None)]*(n+1), method='highs')
    
    if not res.success:
        raise ValueError("Fase I falló. El problema puede ser infactible.")
    
    # Extraer x y t
    x_feasible = res.x[:n]
    t = res.x[-1]
    
    if t > 1e-6:
        print("Advertencia: El problema puede ser infactible (t =", t, ")")
    
    # Multiplicadores duales (lam) de las restricciones de igualdad
    lam = res.eqlin.marginals  # Usar 'eqlin.marginals' en SciPy >=1.6.0
    
    # Holguras duales iniciales (s = c - A^T @ lam)
    s = np.maximum(c - A.T @ lam, 1e-6)  # Asegurar s > 0
    
    # Verificar que x y s sean positivos
    # if not (np.all(x_feasible > 0) and np.all(s > 0)):
    #     raise ValueError("El punto inicial no es factible o las holguras duales no son positivas.")
    
    return x_feasible, lam, s
    
def primal_dual_interior_point(A, b, c, tol=1e-8, max_iter=100, mu_factor=0.1):
    x, lam, s = find_initial_point(A, b, c)
    m, n = A.shape
    history = {'mu': [], 'residual_primal': [], 'residual_dual': []}
    
    for it in range(max_iter):
        mu = np.dot(x, s) / n
        history['mu'].append(mu)
        
        # Calcular residuales
        residual_primal = np.linalg.norm(A @ x - b)
        residual_dual = np.linalg.norm(A.T @ lam + s - c)
        history['residual_primal'].append(residual_primal)
        history['residual_dual'].append(residual_dual)
        
        # Verificar convergencia
        if mu < tol and residual_primal < tol and residual_dual < tol:
            print(f"Convergencia en iteración {it}: μ = {mu:.3e}, residual_primal = {residual_primal:.3e}, residual_dual = {residual_dual:.3e}")
            break
        
        # Resolver sistema de Newton
        dx, dlam, ds = solve_newton_system(A, x, lam, s, b, c, mu)
        
        # Búsqueda de línea con backtracking mejorado
        alpha = 1.0
        max_backtrack = 50  # Aumentar el número máximo de intentos
        for _ in range(max_backtrack):
            x_new = x + alpha * dx
            s_new = s + alpha * ds
            if np.all(x_new > 0) and np.all(s_new > 0):
                break
            alpha *= 0.7  # Reducir el factor de reducción
        else:
            print("Búsqueda de línea fallida. Ajustando dirección de Newton.")
            alpha = 0.0  # No actualizar variables
        
        # Actualizar variables
        x += alpha * dx
        lam += alpha * dlam
        s += alpha * ds
        
        # Reducción de μ en cada iteración
        mu *= 0.5  # Reducir μ en un factor constante
        
        # Verificar convergencia
        if mu < 1e-6 and residual_primal < 1e-6 and residual_dual < 1e-6:
            print(f"Convergencia en iteración {it}: μ = {mu:.3e}, residual_primal = {residual_primal:.3e}, residual_dual = {residual_dual:.3e}")
            break
        
        print(f"Iteración {it}: μ = {mu:.3e}, α = {alpha:.3f}, residual_primal = {residual_primal:.3e}, residual_dual = {residual_dual:.3e}")
    
    else:
        print(f"No convergió en {max_iter} iteraciones. Último μ: {mu:.3e}")
    
    return x, lam, s, history