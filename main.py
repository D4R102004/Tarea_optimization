"""
main.py
------------------------------------
Archivo principal para ejecutar el flujo completo:
1. Corre los experimentos de optimización
2. Analiza los resultados generados
------------------------------------
"""

import os
import experimentos
import analizar_resultados
from graficar import plot_surface
from funcion.function import func
from graficar import grafica_trayectorias 

if __name__ == "__main__":
    print("🚀 INICIANDO EXPERIMENTOS...")
    experimentos.correr_experimentos()

    print("\n🔍 ANALIZANDO RESULTADOS...")
    if os.path.exists("resultados.json"):
        analizar_resultados.__main__ = None  # Evita ejecución duplicada en imports
        analizar_resultados.experimentos = analizar_resultados.cargar_resultados("resultados.json")
        if len(analizar_resultados.experimentos) == 0:
            print("❌ No hay resultados para analizar.")
        else:
            stats = analizar_resultados.calcular_estadisticas(analizar_resultados.experimentos)
            print("\n📊 RESUMEN ESTADÍSTICO (desde main.py):")
            for alg, s in stats.items():
                print(f"\n➡️ {alg}")
                print(f"   Iteraciones promedio: {s['prom_iteraciones']:.2f} ± {s['std_iteraciones']:.2f}")
                print(f"   Tiempo promedio:      {s['prom_tiempo']:.4f} ± {s['std_tiempo']:.4f}")
                print(f"   Valor final promedio: {s['prom_valor_final']:.6e}")

            analizar_resultados.graficar_comparaciones(stats)
    else:
        print("⚠️ No se encontró 'resultados.json'. Ejecuta primero los experimentos.")

    print("\n📈 GENERANDO GRÁFICO DE LA FUNCIÓN...")
    plot_surface(func, x_range=(-2,2), y_range=(-2,2), resolution=300,
                save_path='graficos/grafico_funcion.png', show=False)
    print("\nGENERANDO GRÁFICO DE LA FUNCIÓN CON METODOS...")
    grafica_trayectorias.correr_grafico()



