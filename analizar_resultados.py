"""
analizar_resultados.py
----------------------------------
Este módulo carga los resultados de los experimentos guardados en 'resultados.json',
calcula estadísticas comparativas entre algoritmos y genera gráficos ilustrativos.

Se espera que los resultados tengan el siguiente formato (ver 'guardar_resultados.py'):

[
    {
        "algoritmo": "Gradiente Descendente",
        "punto_inicial": [1.0, -2.0],
        "alpha": 0.05,
        "resultado": {
            "minimo": [0.1, 0.3],
            "valor": 0.001,
            "iteraciones": 120,
            "tiempo": 0.052
        }
    },
    {
        "algoritmo": "Newton",
        ...
    }
]

"""

# =====================================================
# IMPORTACIONES
# =====================================================
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def cargar_resultados(filename="resultados.json"):
    """
    Carga los resultados de un archivo JSON.
    Si el archivo no existe o está vacío, retorna una lista vacía.
    """
    if not os.path.exists(filename):
        print(f"⚠️ No se encontró el archivo '{filename}'.")
        return []
    
    try:
        with open(filename, "r") as f:
            datos = json.load(f)
        print(f"✅ Se cargaron {len(datos)} experimentos desde '{filename}'")
        return datos
    except json.JSONDecodeError:
        print(f"⚠️ El archivo '{filename}' está vacío o corrupto.")
        return []


def agrupar_por_algoritmo(experimentos):
    """
    Agrupa los experimentos por el nombre del algoritmo.
    Devuelve un diccionario donde cada clave es un algoritmo.
    """
    agrupados = defaultdict(list)
    for exp in experimentos:
        agrupados[exp["algoritmo"]].append(exp)
    return agrupados


def calcular_estadisticas(experimentos):
    """
    Calcula estadísticas básicas (media y desviación estándar)
    de número de iteraciones y tiempo de ejecución por algoritmo.
    """
    resultados = {}
    agrupados = agrupar_por_algoritmo(experimentos)

    for alg, lista in agrupados.items():
        iteraciones = [exp["resultado"]["iteraciones"] for exp in lista]
        tiempos = [exp["resultado"]["tiempo"] for exp in lista]
        valores = [exp["resultado"]["valor"] for exp in lista]

        resultados[alg] = {
            "prom_iteraciones": np.mean(iteraciones),
            "std_iteraciones": np.std(iteraciones),
            "prom_tiempo": np.mean(tiempos),
            "std_tiempo": np.std(tiempos),
            "prom_valor_final": np.mean(valores)
        }

    return resultados


# =====================================================
# GRAFICACIÓN DE RESULTADOS
# =====================================================

def graficar_comparaciones(estadisticas, carpeta="graficos"):
    """
    Crea y guarda gráficos comparativos de desempeño entre algoritmos.
    - Tiempo promedio de ejecución
    - Iteraciones promedio
    - Valor final alcanzado
    """

    # Crear carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    algoritmos = list(estadisticas.keys())
    prom_tiempos = [estadisticas[a]["prom_tiempo"] for a in algoritmos]
    prom_iters = [estadisticas[a]["prom_iteraciones"] for a in algoritmos]
    prom_valores = [estadisticas[a]["prom_valor_final"] for a in algoritmos]

    # --- Gráfico 1: Tiempo ---
    plt.figure()
    plt.bar(algoritmos, prom_tiempos, color="skyblue")
    plt.title("Tiempo promedio (s)")
    plt.ylabel("Segundos")
    plt.savefig(f"{carpeta}/tiempos.png", dpi=300)
    plt.close()

    # --- Gráfico 2: Iteraciones ---
    plt.figure()
    plt.bar(algoritmos, prom_iters, color="lightgreen")
    plt.title("Iteraciones promedio")
    plt.ylabel("Iteraciones")
    plt.savefig(f"{carpeta}/iteraciones.png", dpi=300)
    plt.close()

    # --- Gráfico 3: Valor final ---
    plt.figure()
    plt.bar(algoritmos, prom_valores, color="salmon")
    plt.title("Valor final promedio f(x)")
    plt.ylabel("Valor de la función")
    plt.savefig(f"{carpeta}/valores_finales.png", dpi=300)
    plt.close()

    print("📁 Gráficas guardadas en la carpeta 'graficos/'")


# =====================================================
# PROGRAMA PRINCIPAL
# =====================================================

if __name__ == "__main__":
    # 1️⃣ Cargar los experimentos desde el archivo JSON
    experimentos = cargar_resultados("resultados.json")

    if len(experimentos) == 0:
        print("❌ No hay datos para analizar.")
        exit()

    # 2️⃣ Calcular estadísticas globales
    estadisticas = calcular_estadisticas(experimentos)

    # 3️⃣ Mostrar resultados en consola
    print("\n📊 RESUMEN ESTADÍSTICO:")
    for alg, stats in estadisticas.items():
        print(f"\n➡️ {alg}")
        print(f"   Iteraciones promedio: {stats['prom_iteraciones']:.2f} ± {stats['std_iteraciones']:.2f}")
        print(f"   Tiempo promedio:      {stats['prom_tiempo']:.4f} ± {stats['std_tiempo']:.4f}")
        print(f"   Valor final promedio: {stats['prom_valor_final']:.6e}")

    # 4️⃣ Graficar comparaciones visuales
    graficar_comparaciones(estadisticas)
