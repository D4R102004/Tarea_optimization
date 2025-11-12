"""
guardar_resultados.py
----------------------------------
Este módulo se encarga de guardar los resultados de los experimentos
en un archivo JSON llamado 'resultados.json'.

Si el archivo ya existe, los nuevos resultados se añaden al final.
Si no existe, se crea desde cero.

El formato del archivo será una lista de diccionarios, donde cada
diccionario representa un experimento.

"""

import json
import os

# -------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -------------------------------------------------------
def guardar_experimento(resultado, filename="resultados.json"):
    """
    Guarda un experimento individual en un archivo JSON.

    Parámetros
    ----------
    resultado : dict
        Diccionario con los datos del experimento. Ejemplo:
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
        }
    filename : str, opcional
        Nombre del archivo donde se guardarán los resultados (default = 'resultados.json').

    Funcionamiento
    --------------
    1. Si el archivo no existe, crea uno nuevo con el experimento.
    2. Si el archivo existe, lee los datos previos y añade el nuevo experimento.
    3. Maneja errores de lectura/escritura de forma segura.
    """

    try:
        # Verificar si el archivo existe
        if os.path.exists(filename):
            # Cargar los datos existentes
            with open(filename, "r") as f:
                try:
                    datos = json.load(f)
                except json.JSONDecodeError:
                    # Si el archivo está vacío o corrupto, empezar desde cero
                    datos = []
        else:
            datos = []

        # Añadir el nuevo resultado
        datos.append(resultado)

        # Guardar de nuevo todo el contenido
        with open(filename, "w") as f:
            json.dump(datos, f, indent=4)

        print(f"✅ Resultado guardado correctamente en '{filename}'")

    except Exception as e:
        print(f"⚠️ Error al guardar el experimento: {e}")
