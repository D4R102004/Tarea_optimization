# algoritmos/newton.py (mejorado)
import numpy as np
from autograd import grad, hessian
import time

def newton_method(f, x0, tol=1e-6, max_iter=200, mu0=1e-6, verbose=False):
    grad_f = grad(f)
    hess_f = hessian(f)
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    t0 = time.time()

    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        if not np.all(np.isfinite(g)) or not np.all(np.isfinite(H)):
            if verbose: print("Non-finite g or H, stopping.")
            break
        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            if verbose: print(f"Convergencia en iter {k}")
            break

        mu = mu0
        success = False
        # intentar resolver con regularización creciente si singular
        for attempt in range(10):
            try:
                # regularize Hessian
                H_reg = H + mu * np.eye(len(x))
                d = -np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                mu *= 10
                continue

            # damping: backtracking on step length
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
                    print(f"Iter {k}: f={f_new:.6e}, ||g||={norm_g:.3e}, alpha={alpha:.2e}, mu={mu:.1e}")
                break
            else:
                mu *= 10

        if not success:
            if verbose: print("Newton failed to find descent; stopping.")
            break

    elapsed = time.time() - t0
    return x, history, f(x), elapsed
