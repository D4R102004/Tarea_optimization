# Tare_Evaluativa/function.py
import numpy as np

def func(X):
    """
    Función objetivo f(x, y) = (2x^3*y - y^3)^2 + x^2
    X: array-like [x, y]
    Retorna valor escalar f(x, y)
    """
    x, y = X[0], X[1]
    return (2 * x**3 * y - y**3)**2 + x**2