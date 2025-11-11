# Tare_Evaluativa/graph.py
import numpy as np
import matplotlib.pyplot as plt

def plot_surface(func, x_range=(-2, 2), y_range=(-2, 2), resolution=200, cmap='viridis'):
    """
    Grafica la superficie 3D de una función de 2 variables.

    Parámetros:
    - func: función que recibe un array [x, y] y retorna f(x, y)
    - x_range: tupla (min, max) para el eje x
    - y_range: tupla (min, max) para el eje y
    - resolution: número de puntos en cada eje
    - cmap: colormap de matplotlib
    """
    # Crear malla de puntos
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluar función en cada punto
    Z = np.array([[func([xi, yi]) for xi in x] for yi in y])

    # Graficar
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show()
