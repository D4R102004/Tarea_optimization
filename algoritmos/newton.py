"""
newton.py
-------------------------------------
Implementación del método de Newton con regularización (damping y backtracking).

El método de Newton utiliza la información de segunda derivada (Hessiana)
para calcular pasos de actualización más informados, lo que suele garantizar
una convergencia cuadrática cerca del mínimo.

Incluye estrategias de:
- Regularización de la Hessiana (mu * I)
- Backtracking en el tamaño del paso
- Manejo de errores numéricos

Proyecto: Tarea Evaluativa - Optimización Numérica
"""

import numpy as np
from autograd import grad, hessian
import time

# =====================================================
# MÉTODO DE NEWTON
# =====================================================

def newton_method(f, x0, tol=1e-6, max_iter=200, mu0=1e-6, verbose=False):
    """
    Implementación robusta del método de Newton para optimización.

    Parámetros:
    ------------
    f : callable
        Función objetivo a minimizar.
    x0 : array-like
        Punto inicial (vector).
    tol : float
        Tolerancia para el gradiente.
    max_iter : int
        Número máximo de iteraciones.
    mu0 : float
        Parámetro inicial de regularización.
    verbose : bool
        Si True, imprime información iterativa.

    Retorna:
    --------
    x : np.ndarray
        Aproximación del punto mínimo encontrado.
    history : list
        Secuencia de puntos visitados durante el proceso.
    f(x) : float
        Valor de la función en el mínimo encontrado.
    elapsed : float
        Tiempo total de ejecución.
    """

    grad_f = grad(f)
    hess_f = hessian(f)
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    t0 = time.time()

    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)

        # Validación numérica
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(H)):
            if verbose: print("Gradiente o Hessiana no finitos. Deteniendo.")
            break

        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            if verbose: print(f"Convergencia alcanzada en iteración {k}")
            break

        mu = mu0
        success = False

        # =====================================================
        # Regularización + backtracking adaptativo
        # =====================================================
        for attempt in range(10):
            try:
                # Regularización de la Hessiana (para evitar singularidades)
                H_reg = H + mu * np.eye(len(x))
                d = -np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                mu *= 10  # Incrementar regularización si falla
                continue

            # Búsqueda de línea: reducir paso si no mejora
            alpha = 1.0
            f_x = f(x)
            for _ in range(30):
                x_new = x + alpha * d
                f_new = f(x_new)
                if not np.isfinite(f_new):
                    alpha *= 0.5
                    continue
                if f_new <= f_x + 1e-4 * alpha * np.dot(g, d):
                    success = True
                    break
                alpha *= 0.5

            if success:
                x = x_new
                history.append(x.copy())
                if verbose:
                    print(f"Iter {k}: f={f_new:.6e}, ||g||={norm_g:.3e}, α={alpha:.2e}, μ={mu:.1e}")
                break
            else:
                mu *= 10  # Aumentar regularización si no hay descenso

        if not success:
            if verbose: print("No se encontró dirección de descenso. Deteniendo.")
            break

    elapsed = time.time() - t0
    return x, history, f(x), elapsed
