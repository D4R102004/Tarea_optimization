from funcion.function import func
from algoritmos.gradiente_descendente import gradiente_descendente
from algoritmos.newton import newton_method
from graficar.grafico import plot_surface

import numpy as np

if __name__ == "__main__":
    print("==== OPTIMIZACIÓN ====")

    x0 = np.array([1.0, -1.0])  # punto inicial
    print("Punto inicial:", x0)

    # Gradiente Descendente
    x_gd, hist_gd = gradiente_descendente(func, x0, verbose=True)
    print(f"\nGradiente Descendente → mínimo aprox en {x_gd}, f(x) = {func(x_gd):.6f}")

    # Newton
    x_newton, hist_newton = newton_method(func, x0, verbose=True)
    print(f"\nNewton → mínimo aprox en {x_newton}, f(x) = {func(x_newton):.6f}")

    # Gráfico de la función base
    plot_surface(func, x_range=(-2, 2), y_range=(-2, 2), resolution=200, cmap='viridis')
