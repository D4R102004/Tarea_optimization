# algoritmos/gradiente_descendente.py (mejorado)
import numpy as np
from autograd import grad
import time

def safe_eval(f, x):
    val = f(x)
    if not np.isfinite(val):
        return np.inf
    return val

def gradiente_descendente(f, x0, alpha0=1.0, alpha_max=1e2, rho=0.5, m1=1e-4,
                          tol=1e-6, max_iter=5000, verbose=False):
    grad_f = grad(f)
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    f_x = safe_eval(f, x)
    t0 = time.time()

    for k in range(max_iter):
        g = grad_f(x)
        if not np.all(np.isfinite(g)):
            if verbose: print("grad non-finite, stopping.")
            break
        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            if verbose: print(f"Convergencia en iter {k}")
            break

        d = -g
        alpha = min(alpha0, alpha_max)
        # Armijo backtracking with max reductions
        reductions = 0
        while True:
            x_new = x + alpha * d
            f_new = safe_eval(f, x_new)
            if f_new <= f_x + m1 * alpha * np.dot(g, d):
                break
            alpha *= rho
            reductions += 1
            if alpha < 1e-12 or reductions > 60 or not np.isfinite(alpha):
                # fallback: step proportional to gradient (very small)
                alpha = 1e-8
                x_new = x + alpha * d
                f_new = safe_eval(f, x_new)
                break

        x = x_new
        f_x = f_new
        history.append(x.copy())
        if verbose and k % 10 == 0:
            print(f"Iter {k}: f={f_x:.6e}, ||g||={norm_g:.3e}, alpha={alpha:.2e}")

    elapsed = time.time() - t0
    return x, history, f_x, elapsed
