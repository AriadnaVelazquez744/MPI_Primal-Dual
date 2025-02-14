import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def primal_dual_interior_point(A, b, c, x0, y0, s0, tol=1e-6, max_iter=100, tau=0.95):
    m, n = A.shape
    x = x0.copy()
    y = y0.copy()
    s = s0.copy()
    history = {'x': [], 'y': [], 's': [], 'mu': [], 'r_b': [], 'r_c': []}
    eps = 1e-12  # Regularización para evitar divisiones por cero

    for it in range(max_iter):
        # Residuales y medida de complementariedad
        r_b = A @ x - b
        r_c = A.T @ y + s - c
        mu = np.dot(x, s) / n

        # Guardar historia
        history['x'].append(x.copy())
        history['y'].append(y.copy())
        history['s'].append(s.copy())
        history['mu'].append(mu)
        history['r_b'].append(np.linalg.norm(r_b))
        history['r_c'].append(np.linalg.norm(r_c))

        # Depuración
        print(f"\nIteración {it}:")
        print(f"  x = {x}")
        print(f"  s = {s}")
        print(f"  μ = {mu:.6f}")
        print(f"  ||r_b|| = {np.linalg.norm(r_b):.6f}")
        print(f"  ||r_c|| = {np.linalg.norm(r_c):.6f}")

        # Criterio de parada
        if (np.linalg.norm(r_b) < tol and np.linalg.norm(r_c) < tol and mu < tol):
            print("\nConvergencia alcanzada.")
            break

        # Matriz de precondicionamiento
        X_inv = np.diag(1 / np.maximum(x, eps))
        S = np.diag(s)
        Theta = X_inv @ S
        M = A @ (Theta @ A.T) + eps * np.eye(m)  # Regularización

        # Dirección predictor (affine)
        rhs_aff = A @ (Theta @ r_c - x) + r_b
        dy_aff = np.linalg.solve(M, rhs_aff)
        dx_aff = Theta @ (A.T @ dy_aff - r_c) + x
        ds_aff = -r_c - A.T @ dy_aff

        # Paso affine
        alpha_p_aff = min(tau * np.min(-x[dx_aff < 0] / dx_aff[dx_aff < 0]), 1) if np.any(dx_aff < 0) else 1
        alpha_d_aff = min(tau * np.min(-s[ds_aff < 0] / ds_aff[ds_aff < 0]), 1) if np.any(ds_aff < 0) else 1

        # Medida de complementariedad affine
        mu_aff = np.dot(x + alpha_p_aff * dx_aff, s + alpha_d_aff * ds_aff) / n
        sigma = (mu_aff / mu) ** 3

        # Dirección corrector-centrante
        rhs_cc = A @ (Theta @ (r_c + sigma * mu * np.ones(n)) - x) + r_b
        dy_cc = np.linalg.solve(M, rhs_cc)
        dx_cc = Theta @ (A.T @ dy_cc - (r_c + sigma * mu * np.ones(n))) + x
        ds_cc = -(r_c + sigma * mu * np.ones(n)) - A.T @ dy_cc

        # Paso completo
        alpha_p = min(tau * np.min(-x[dx_cc < 0] / dx_cc[dx_cc < 0]), 1) if np.any(dx_cc < 0) else 1
        alpha_d = min(tau * np.min(-s[ds_cc < 0] / ds_cc[ds_cc < 0]), 1) if np.any(ds_cc < 0) else 1

        # Depuración de pasos
        print(f"  dx_aff = {dx_aff}")
        print(f"  ds_aff = {ds_aff}")
        print(f"  dx_cc = {dx_cc}")
        print(f"  ds_cc = {ds_cc}")
        print(f"  alpha_p = {alpha_p:.6f}, alpha_d = {alpha_d:.6f}")

        # Actualización
        x += alpha_p * dx_cc
        y += alpha_d * dy_cc
        s += alpha_d * ds_cc

        # Garantizar positividad
        x = np.maximum(x, eps)
        s = np.maximum(s, eps)

    return x, y, s, history

# Configuración del problema
A = np.array([[1, 1], [1, -1]], dtype=float)
b = np.array([2, 0], dtype=float)
c = np.array([-1, -1], dtype=float)

# Punto inicial ajustado
x0 = np.array([1.5, 0.5])
y0 = np.array([-1.0, 0.0])
s0 = np.maximum(c - A.T @ y0, 0.1)

# Resolver con método corregido
x_pd, y_pd, s_pd, history = primal_dual_interior_point(A, b, c, x0, y0, s0)

# Comparación con Simplex
result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None))

# Resultados
print("\nPrimal-Dual Corregido:")
print(f"Solución x: {x_pd}")
print(f"Valor objetivo: {c @ x_pd:.4f}")

print("\nSimplex:")
print(f"Solución x: {result.x}")
print(f"Valor objetivo: {result.fun:.4f}")

# Gráficas de convergencia
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.semilogy(history['mu'])
plt.title('Convergencia de μ')
plt.subplot(1, 3, 2)
plt.semilogy(history['r_b'])
plt.title('Residual Primal')
plt.subplot(1, 3, 3)
plt.semilogy(history['r_c'])
plt.title('Residual Dual')
plt.tight_layout()
plt.show()