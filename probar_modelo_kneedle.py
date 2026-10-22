import argparse
import json
import re
import numpy as np
import pandas as pd
from scipy import sparse

from modelo_unificado import separar_prueba
from comparativa_kmeans_propio import METRICAS

def nombres_clusters(centroides, metrica_nombre, grupo, X):
    metrica = METRICAS[metrica_nombre]()
    Z = metrica.format_data(X)
    distancias = metrica.get_distance(Z, centroides)
    etiquetas = np.argmin(distancias, axis=1)
    
    nombres = {}
    for c in range(len(centroides)):
        sentidos = grupo["sentido"].to_numpy()[etiquetas == c]
        if len(sentidos):
            valores, cuentas = np.unique(sentidos, return_counts=True)
            nombres[c] = (valores[cuentas.argmax()], cuentas.max(), len(sentidos))
        else:
            nombres[c] = ("(vacío)", 0, 0)
    return nombres, etiquetas

def main():
    parser = argparse.ArgumentParser(description="Prueba los modelos .json generados por train.py")
    parser.add_argument("--abreviatura", required=True)
    parser.add_argument("--modelo", required=True, help="Ruta al .json (ej. modelos_finales/CA_k6.json)")
    parser.add_argument("--matriz", default="representaciones/X_bert_abrev.npy")
    parser.add_argument("--instancias", default="representaciones/instancias.csv")
    parser.add_argument("--metrica", default="coseno", choices=["coseno", "euclidea"])
    parser.add_argument("--ejemplos", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    abrev = args.abreviatura
    
    # 1. Cargar centroides de tu archivo JSON
    with open(args.modelo, 'r') as f:
        datos_modelo = json.load(f)
    
    clave_centroides = next((k for k in ["centroids", "centroides"] if k in datos_modelo), None)
    if clave_centroides is None:
        raise ValueError("No se encontraron centroides en el JSON.")
    centroides = np.array(datos_modelo[clave_centroides])

    # 2. Cargar datos base
    df = pd.read_csv(args.instancias, usecols=["abreviatura", "sentido", "texto"])
    grupo = df[df["abreviatura"] == abrev]
    
    if args.matriz.endswith(".npz"):
        M_global = sparse.load_npz(args.matriz)
    else:
        M_global = np.load(args.matriz, mmap_mode="r")
        
    # 3. Replicar la MISMA separación de entrenamiento y prueba usando la semilla
    rng = np.random.default_rng(args.semilla)
    for a, g in df.groupby("abreviatura"):
        es_prueba = separar_prueba(g, 0.2, rng)
        if a == abrev:
            break
            
    entrena, prueba = grupo[~es_prueba], grupo[es_prueba]
    
    # 4. Extraer vectores
    def extraer_vectores(indices):
        X = M_global[indices]
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)
        
    X_entrena = extraer_vectores(entrena.index.to_numpy())
    
    # 5. Analizar clusters (nombrarlos según entrenamiento)
    nombres, _ = nombres_clusters(centroides, args.metrica, entrena, X_entrena)
    
    print(f"\n{abrev}: {len(centroides)} clusters analíticos (Kneedle) ({grupo['sentido'].nunique()} sentidos reales)")
    for c, (sentido, n, total) in nombres.items():
        print(f"  cluster {c}: predomina '{sentido}' ({n} de sus {total} textos de entrenamiento)")

    # 6. Predecir sobre los mismos ejemplos exactos de prueba
    muestra = prueba.sample(min(args.ejemplos, len(prueba)), random_state=args.semilla)
    X_muestra = extraer_vectores(muestra.index.to_numpy())
    
    _, etiquetas_muestra = nombres_clusters(centroides, args.metrica, muestra, X_muestra)
    
    coinciden = 0
    print(f"\n{len(muestra)} contextos de prueba (no vistos al entrenar):")
    for (_, fila), c in zip(muestra.iterrows(), etiquetas_muestra):
        coincide = nombres[c][0] == fila["sentido"]
        coinciden += coincide
        texto = re.sub(r"\s+", " ", fila["texto"])
        print(f"\n  {'✔' if coincide else '✘'} asignado al cluster {c} (predomina '{nombres[c][0]}')  |  sentido real: {fila['sentido']}")
        print(f"    ...{texto[:160]}...")
        
    print(f"\nSentido real = sentido predominante de su cluster en {coinciden}/{len(muestra)} textos")

if __name__ == "__main__":
    main()
