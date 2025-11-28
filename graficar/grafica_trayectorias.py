"""
grafica_trayectorias.py
-------------------------------------
Genera una gráfica comparando la trayectoria del
Gradiente Descendente y del Método de Newton sobre
contornos de la función objetivo.

Se usa un mismo punto inicial para visualizar claramente
cómo se comportan los métodos.
"""

import numpy as np
import matplotlib.pyplot as plt
from funcion.function import func
from algoritmos.gradiente_descendente import gradiente_descendente
from algoritmos.newton import newton_method
import os

# =====================================================
# FUNCIÓN AUXILIAR
# =====================================================

def extraer_trayectoria(history):
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return xs, ys

# =====================================================
# GRAFICAR TRAYECTORIAS
# =====================================================

def correr_grafico():
    print("📈 Generando gráfica de trayectorias...")

    # Punto inicial recomendado (no explota la función)
    x0 = np.array([2.0, 2.0])

    # Ejecutar GD
    x_gd, hist_gd, f_gd, t_gd = gradiente_descendente(
        func, x0, alpha0=0.05, max_iter=400
    )
    
    # Ejecutar Newton
    x_nw, hist_nw, f_nw, t_nw = newton_method(
        func, x0, max_iter=50
    )

    # Convertir trayectoria a listas para graficar
    xs_gd, ys_gd = extraer_trayectoria(hist_gd)
    xs_nw, ys_nw = extraer_trayectoria(hist_nw)

    # -------------------------
    # Crear el mapa de contorno
    # -------------------------
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    Z = func(np.array([X, Y]))

    # -------------------------
    # GRAFICAR
    # -------------------------
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=40)

    # Trayectorias
    plt.plot(xs_gd, ys_gd, 'o-', markersize=3, label="Gradiente Descendente")
    plt.plot(xs_nw, ys_nw, 'x--', markersize=4, label="Newton")

    plt.scatter([x0[0]], [x0[1]], color="red", label="Punto inicial")

    plt.title("Trayectoria de los métodos de optimización")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Asegurar carpeta graficos/
    if not os.path.exists("graficos"):
        os.makedirs("graficos")

    path = "graficos/trayectoria_algoritmos.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"📌 Gráfica guardada en: {path}")
