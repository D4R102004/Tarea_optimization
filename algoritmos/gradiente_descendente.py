"""
gradiente_descendente.py
-------------------------------------
Implementación del método de Gradiente Descendente con búsqueda de línea adaptativa
usando la condición de Armijo (backtracking line search).

Este algoritmo aproxima un mínimo local de una función diferenciable f(x),
moviendo el punto actual en la dirección opuesta al gradiente (descenso más pronunciado).

Autor: Sakaki
Proyecto: Tarea Evaluativa - Optimización Numérica
"""

import numpy as np
from autograd import grad
import time

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def safe_eval(f, x):
    """
    Evalúa f(x) y maneja valores no finitos devolviendo np.inf.
    Evita que el programa colapse si la función explota numéricamente.
    """
    val = f(x)
    return val if np.isfinite(val) else np.inf


# =====================================================
# MÉTODO DE GRADIENTE DESCENDENTE
# =====================================================

def gradiente_descendente(f, x0, alpha0=1.0, alpha_max=1e2, rho=0.5, m1=1e-4,
                          tol=1e-6, max_iter=5000, verbose=False):
    """
    Implementación general del método de gradiente descendente con búsqueda de línea (Armijo).

    Parámetros:
    ------------
    f : callable
        Función objetivo a minimizar.
    x0 : array-like
        Punto inicial (vector).
    alpha0 : float
        Paso inicial para la búsqueda de línea.
    alpha_max : float
        Máximo tamaño de paso permitido.
    rho : float
        Factor de reducción del paso durante el backtracking (0 < rho < 1).
    m1 : float
        Constante de Armijo (usualmente pequeña, e.g. 1e-4).
    tol : float
        Tolerancia para el criterio de parada basado en ||grad f(x)||.
    max_iter : int
        Máximo número de iteraciones.
    verbose : bool
        Si True, imprime información de cada iteración.

    Retorna:
    --------
    x : np.ndarray
        Aproximación del punto mínimo encontrado.
    history : list
        Secuencia de puntos visitados durante el proceso.
    f_x : float
        Valor de la función en el mínimo encontrado.
    elapsed : float
        Tiempo total de ejecución (segundos).
    """

    grad_f = grad(f)                     # Gradiente automático
    x = np.array(x0, dtype=float)        # Copia inicial
    f_x = safe_eval(f, x)
    history = [x.copy()]                 # Guardar trayectoria
    t0 = time.time()                     # Medir tiempo de ejecución

    for k in range(max_iter):
        g = grad_f(x)
        if not np.all(np.isfinite(g)):
            if verbose: print("Gradiente no finito. Deteniendo.")
            break

        # Criterio de convergencia
        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            if verbose: print(f"Convergencia alcanzada en iteración {k}")
            break

        # Dirección de descenso
        d = -g
        alpha = min(alpha0, alpha_max)

        # =====================================================
        # Búsqueda de línea con condición de Armijo
        # =====================================================
        reductions = 0
        while True:
            x_new = x + alpha * d
            f_new = safe_eval(f, x_new)
            # Condición de Armijo: mejora suficiente
            if f_new <= f_x + m1 * alpha * np.dot(g, d):
                break
            alpha *= rho
            reductions += 1
            if alpha < 1e-12 or reductions > 60:
                # Evita pasos diminutos o bucles infinitos
                alpha = 1e-8
                x_new = x + alpha * d
                f_new = safe_eval(f, x_new)
                break

        # Actualización de variables
        x = x_new
        f_x = f_new
        history.append(x.copy())

        if verbose and k % 10 == 0:
            print(f"Iter {k}: f={f_x:.6e}, ||g||={norm_g:.3e}, α={alpha:.2e}")

    elapsed = time.time() - t0
    return x, history, f_x, elapsed
