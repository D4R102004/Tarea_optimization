import json
import numpy as np
from funcion.function import func
from algoritmos.gradiente_descendente import gradiente_descendente
from algoritmos.newton import newton_method
from algoritmos.guardar_resultados import guardar_experimento

# ----------------------------------------------
# CONFIGURACIONES DE EXPERIMENTO
# ----------------------------------------------
puntos_iniciales = [
    np.array([2.0, -3.0]),
    np.array([-1.0, 1.5]),
    np.array([0.5, 0.5])
]

alphas = [0.01, 0.05, 0.1]
max_iter = 200

# ----------------------------------------------
# EJECUCIÓN DE EXPERIMENTOS
# ----------------------------------------------
def correr_experimentos():
    resultados = []

    for punto in puntos_iniciales:
        for alpha in alphas:
            # ---- Gradiente Descendente ----
            try:
                x_opt, history, f_opt, tiempo = gradiente_descendente(
                    func, punto, alpha0=alpha, max_iter=max_iter
                )
                resultado_gd = {
                    "algoritmo": "Gradiente Descendente",
                    "punto_inicial": punto.tolist(),
                    "alpha": alpha,
                    "max_iter": max_iter,
                    "resultado": {
                        "minimo": x_opt.tolist(),
                        "valor": float(f_opt),
                        "iteraciones": len(history),
                        "tiempo": tiempo,
                        "convergio": True
                    }
                }
                resultados.append(resultado_gd)
                guardar_experimento(resultado_gd)
            except Exception as e:
                resultados.append({
                    "algoritmo": "Gradiente Descendente",
                    "punto_inicial": punto.tolist(),
                    "alpha": alpha,
                    "error": str(e)
                })

            # ---- Método de Newton ----
            try:
                x_opt, history, f_opt, tiempo = newton_method(
                    func, punto, max_iter=max_iter
                )
                resultado_newton = {
                    "algoritmo": "Newton",
                    "punto_inicial": punto.tolist(),
                    "alpha": "auto",
                    "max_iter": max_iter,
                    "resultado": {
                        "minimo": x_opt.tolist(),
                        "valor": float(f_opt),
                        "iteraciones": len(history),
                        "tiempo": tiempo,
                        "convergio": True
                    }
                }
                resultados.append(resultado_newton)
                guardar_experimento(resultado_newton)
            except Exception as e:
                resultados.append({
                    "algoritmo": "Newton",
                    "punto_inicial": punto.tolist(),
                    "error": str(e)
                })

    # Guardar archivo completo
    with open("resultados_completos.json", "w") as f:
        json.dump(resultados, f, indent=4)

    print("\n✅ Experimentos completados y guardados en 'resultados_completos.json'.")
    print("También se guardaron individualmente en 'resultados.json'.")

# ----------------------------------------------
if __name__ == "__main__":
    correr_experimentos()
