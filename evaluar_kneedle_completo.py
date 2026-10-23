import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import sparse

from modelo_unificado import separar_prueba
from comparativa_kmeans_propio import METRICAS
# Importamos AMBAS funciones de evaluación
from evaluacion import evaluar_con_etiquetas, evaluar_clustering

def main():
    archivo_matriz = "representaciones/X_bert_abrev.npy"
    archivo_instancias = "representaciones/instancias.csv"
    carpeta_modelos = "modelos_finales"
    metrica_nombre = "coseno"
    semilla = 42

    print("Cargando datos y evaluando modelos Kneedle...")
    df = pd.read_csv(archivo_instancias, usecols=["abreviatura", "sentido"])
    if archivo_matriz.endswith(".npz"):
        M = sparse.load_npz(archivo_matriz)
    else:
        M = np.load(archivo_matriz, mmap_mode="r")

    rng = np.random.default_rng(semilla)
    metrica = METRICAS[metrica_nombre]()
    filas = []

    for abrev, grupo in df.groupby("abreviatura"):
        archivos_json = glob.glob(f"{carpeta_modelos}/{abrev}_k*.json")
        if not archivos_json:
            continue
        archivo_json = archivos_json[0] 
        
        with open(archivo_json, 'r') as f:
            datos_modelo = json.load(f)
        
        clave_centroides = next((k for k in ["centroids", "centroides"] if k in datos_modelo), None)
        centroides = np.array(datos_modelo[clave_centroides])
        k_kneedle = len(centroides)

        es_prueba = separar_prueba(grupo, 0.2, rng)
        entrena, prueba = grupo[~es_prueba], grupo[es_prueba]

        fila = {
            "abreviatura": abrev,
            "n_entrena": len(entrena),
            "n_prueba": len(prueba),
            "k_real": grupo["sentido"].nunique(),
            "k_kneedle": k_kneedle
        }

        for nombre, parte in [("entrena", entrena), ("prueba", prueba)]:
            if len(parte) < 2:
                continue
            
            indices = parte.index.to_numpy()
            Xp = M[indices]
            if sparse.issparse(Xp):
                Xp = Xp.toarray()
            Xp = np.asarray(Xp, dtype=np.float64)
            
            Z = metrica.format_data(Xp)
            distancias = metrica.get_distance(Z, centroides)
            etiquetas = np.argmin(distancias, axis=1)

            # 1. Llamada a métricas externas (ARI, NMI, Pureza)
            metricas_ext = evaluar_con_etiquetas(parte["sentido"].to_numpy(), etiquetas)
            fila.update({f"{m}_{nombre}": v for m, v in metricas_ext.items()})

            # 2. Llamada a métricas internas (Silueta, Calinski, Davies)
            # Pasamos Z (datos formateados) para que mida las distancias reales usadas por el modelo
            metricas_int = evaluar_clustering(Z, etiquetas)
            fila.update({
                f"silueta_{nombre}": metricas_int["silhouette_score"],
                f"calinski_harabasz_{nombre}": metricas_int["calinski_harabasz_score"],
                f"davies_bouldin_{nombre}": metricas_int["davies_bouldin_score"]
            })

        filas.append(fila)

    df_resultados = pd.DataFrame(filas)
    df_resultados.round(4).to_csv("kneedle_por_abreviatura_completo.csv", index=False)
    
    print("\nResultados globales guardados en 'kneedle_por_abreviatura_completo.csv'")

if __name__ == "__main__":
    main()
