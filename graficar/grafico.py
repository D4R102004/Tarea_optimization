# Tare_Evaluativa/graph.py
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_surface(func,
                 x_range=(-2, 2),
                 y_range=(-2, 2),
                 resolution=200,
                 cmap='viridis',
                 save_path=None,   # si None => no guarda
                 show=True,        # si True => plt.show()
                 elev=30, azim=-60 # ángulo de cámara opcional
                 ):
    """
    Grafica la superficie 3D de una función de 2 variables y opcionalmente
    guarda la imagen en disco.

    Parámetros
    - func: función que recibe un array [x, y] y retorna f(x, y)
    - x_range, y_range: tuplas (min, max) para los ejes
    - resolution: número de puntos en cada eje (200-400 recomendado)
    - cmap: colormap de matplotlib
    - save_path: ruta de guardado (ej. 'graficos/grafico_funcion.png'). Si None, no guarda.
    - show: si True se muestra la figura en pantalla
    - elev, azim: elevación y azimut para la vista 3D
    """

    # Crear malla de puntos
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluar función en cada punto (vectorizado por simplicidad)
    Z = np.zeros_like(X, dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])

    # Crear la figura 3D
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9)

    # Ajustes de presentación
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='f(x,y)')

    # Guardar antes de mostrar (para evitar problemas con backends)
    if save_path is not None:
        # Crear carpeta si no existe
        folder = os.path.dirname(save_path)
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✅ Gráfico guardado en: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # devolver la figura y el eje por si quieres manipularlos desde el programa
    return fig, ax
