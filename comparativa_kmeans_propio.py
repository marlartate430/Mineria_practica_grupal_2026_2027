"""
Comparativa de representaciones vectoriales usando NUESTRO K-Means (modeloKMeans.py) en vez del de sklearn.

La mejor representación es la que mejor funciona con el clustering que vamos a usar de verdad, así que
cada representación guardada por representacion_vectorial.py se agrupa con nuestro K-Means y se mide
cuánto se parecen los clusters a los sentidos reales con las métricas externas de evaluacion.py (ARI, NMI y
pureza). Las métricas internas (silueta...) no se usan aquí: son para evaluar los clusters del modelo final.

Para cada abreviatura y representación:
    - k = número de sentidos de la abreviatura
    - se lanza el K-Means con varias semillas y se queda la de menor inercia (como n_init de sklearn,
      sin mirar las etiquetas); además se guarda la media y desviación del ARI entre semillas (estabilidad)

Uso (después de ejecutar representacion_vectorial.py):
    python comparativa_kmeans_propio.py
    python comparativa_kmeans_propio.py --metodos tfidf w2v bert_abrev --metrica coseno euclidea
    python comparativa_kmeans_propio.py --init kmeans++ random --n-semillas 20
    python comparativa_kmeans_propio.py --varianza-pca 0.8 0.9 0.95     # prueba varios % de varianza

Resultados en --salida (por defecto 'representaciones/'):
    comparativa_kmeans_propio.csv, comparativa_kmeans_propio_por_abreviatura.csv
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse  # solo para leer las matrices dispersas (.npz) que guarda representacion_vectorial.py

from evaluacion import evaluar_con_etiquetas
from kmeans_init import KMeans_parallel, kmeans_plus_plus, random_cluster_selection
from metrics import CosineMetric, EuclideanMetric
from modeloKMeans import KMeans

# Las piezas de nuestro K-Means: la métrica (metrics.py) y la inicialización (kmeans_init.py)
METRICAS = {"coseno": CosineMetric, "euclidea": EuclideanMetric}
INICIALIZACIONES = {
    "random": lambda: random_cluster_selection,
    "kmeans++": lambda: kmeans_plus_plus,
    "kmeans||": lambda: KMeans_parallel(None, None).initialize,  # factor y rondas por defecto
}


def cargar_matrices(directorio, metodos=None):
    matrices = {}
    for ruta in sorted(glob.glob(f"{directorio}/X_*.npy") + glob.glob(f"{directorio}/X_*.npz")):
        nombre = os.path.basename(ruta)[2:].rsplit(".", 1)[0]
        if metodos and nombre not in metodos:
            continue
        matrices[nombre] = sparse.load_npz(ruta).tocsr() if ruta.endswith(".npz") else np.load(ruta)
    return matrices


# ---------------------------------------------------------------------------
# PCA hecho a mano con numpy (sin sklearn)
# ---------------------------------------------------------------------------

def pca(X, varianza):
    """
    Centra X y la proyecta en sus direcciones de máxima varianza (SVD), quedándose con el mínimo número
    de componentes que explican al menos esa fracción de la varianza (p. ej. 0.9 = 90 %).
    """
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    # La varianza que explica cada componente es proporcional a su valor singular al cuadrado
    acumulada = np.cumsum(S ** 2) / np.sum(S ** 2)
    n_componentes = int(np.searchsorted(acumulada, varianza) + 1)
    return U[:, :n_componentes] * S[:n_componentes]


def inercia(km, X):
    """Suma de distancias de cada punto a su centroide: no usa etiquetas, sirve para elegir semilla."""
    Xf = km.metric.format_data(X)
    return float(km.metric.get_distance(Xf, km.centroids).min(axis=1).sum())


def agrupar(X, y, metrica, init, semillas):
    """Lanza nuestro K-Means con cada semilla y devuelve las métricas de la mejor ejecución."""
    k = len(np.unique(y))
    ejecuciones = []
    for semilla in semillas:
        km = KMeans(p_cluster_number=k, p_metric=METRICAS[metrica](), p_init=INICIALIZACIONES[init]())
        km.fit(X, semilla)
        etiquetas = km.predict(X)
        ejecuciones.append((inercia(km, X), evaluar_con_etiquetas(y, etiquetas)))
    mejor = min(range(len(ejecuciones)), key=lambda i: ejecuciones[i][0])
    aris = np.array([externas["ARI"] for _, externas in ejecuciones])
    return {**ejecuciones[mejor][1], "ARI_medio": aris.mean(), "ARI_std": aris.std()}


def main():
    parser = argparse.ArgumentParser(description="Comparativa de representaciones con nuestro K-Means")
    parser.add_argument("--representaciones", default="representaciones",
                        help="Carpeta con instancias.csv y X_<metodo>.npy/.npz")
    parser.add_argument("--metodos", nargs="+", default=None, help="Por defecto, todas las X_* que haya")
    parser.add_argument("--metrica", nargs="+", default=["coseno"], choices=list(METRICAS))
    parser.add_argument("--init", nargs="+", default=["random"], choices=list(INICIALIZACIONES))
    parser.add_argument("--n-semillas", type=int, default=10)
    parser.add_argument("--varianza-pca", nargs="+", type=float, default=[0.9],
                        help="Fracción de varianza que deben explicar las variantes '+pca' (0 = no calcularlas)")
    parser.add_argument("--salida", default=None, help="Por defecto, la carpeta de --representaciones")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--reanudar", action="store_true",
                        help="Saltarse las combinaciones que ya están en el archivo parcial de --salida")
    args = parser.parse_args()
    salida = args.salida or args.representaciones

    df = pd.read_csv(f"{args.representaciones}/instancias.csv")
    matrices = cargar_matrices(args.representaciones, args.metodos)
    if not matrices:
        raise SystemExit(f"No hay matrices X_* en '{args.representaciones}': ejecuta antes representacion_vectorial.py")
    print(f"{len(df)} instancias, representaciones: {', '.join(matrices)}")

    # Algunas instancias se quedan sin ninguna palabra tras el preprocesado (p.ej. el contexto es solo "CEA") y su
    # vector es todo ceros en BoW/TF-IDF/LSA: con coseno la distancia no está definida (NaN) y el K-Means++ falla.
    # Se quitan de TODAS las representaciones para que todas se comparen con las mismas instancias.
    vacias = np.zeros(len(df), dtype=bool)
    for X in matrices.values():
        vacias |= (X.getnnz(axis=1) == 0) if sparse.issparse(X) else ~np.any(X, axis=1)
    if vacias.any():
        print(f"Se descartan {vacias.sum()} instancias con vector vacío: "
              f"{', '.join(df.loc[vacias, 'abreviatura'].unique())}")
        df = df[~vacias]  # se conserva el índice original, que es la fila en las matrices X

    # Cada representación se prueba tal cual y con PCA. Muchos vectores son anisótropos (todos apuntan en una
    # dirección parecida y el coseno entre cualquier par es alto); centrarlos y reducirlos con PCA dentro de cada
    # abreviatura quita esa componente común. Si ayuda o no a cada método se decide aquí, experimentalmente.
    # Las dimensiones no se fijan a mano: se quedan las necesarias para explicar el % de varianza de --varianza-pca
    # nombre de la variante -> (matriz de origen, fracción de varianza del PCA o None si no se aplica)
    variantes = {nombre: (nombre, None) for nombre in matrices}
    for varianza in args.varianza_pca:
        if 0 < varianza <= 1:
            variantes.update({f"{n}+pca{varianza:.0%}": (n, varianza) for n in matrices})
    semillas = [args.semilla + i for i in range(args.n_semillas)]

    # Cada combinación se guarda en un archivo parcial en cuanto termina: si la ejecución se corta, con --reanudar
    # se continúa desde ahí en vez de empezar de cero
    parcial = f"{salida}/comparativa_kmeans_propio_por_abreviatura_parcial.csv"
    filas, hechas = [], set()
    if args.reanudar and os.path.exists(parcial):
        anteriores = pd.read_csv(parcial)
        filas = anteriores.to_dict("records")
        hechas = set(map(tuple, anteriores[["metodo", "metrica", "init"]].drop_duplicates().values))
        print(f"Se reanuda: {len(hechas)} combinaciones ya hechas en '{parcial}'")
    elif os.path.exists(parcial):
        os.remove(parcial)
    for nombre, (origen, varianza) in variantes.items():
        for metrica in args.metrica:
            for init in args.init:
                if init == "kmeans||" and metrica != "euclidea":
                    continue  # K-Means|| solo está implementado con distancia euclídea
                if (nombre, metrica, init) in hechas:
                    continue
                inicio, n_antes = time.time(), len(filas)
                for abrev, grupo in df.groupby("abreviatura"):
                    X = matrices[origen][grupo.index.to_numpy()]
                    if sparse.issparse(X):
                        # Solo las columnas (palabras) que aparecen en esta abreviatura: el resto son ceros y no
                        # cambian ninguna distancia, pero con todo el vocabulario la matriz densa no cabría en memoria
                        X = X[:, X.getnnz(axis=0) > 0].toarray()
                    X = np.asarray(X, dtype=np.float64)
                    if varianza:
                        X = pca(X, varianza)
                    try:
                        resultado = agrupar(X, grupo["sentido"].values, metrica, init, semillas)
                    except Exception as e:
                        raise SystemExit(f"Nuestro K-Means ha fallado con '{nombre}' ({metrica}, {init}) "
                                         f"en {abrev}: {type(e).__name__}: {e}")
                    filas.append({"metodo": nombre, "metrica": metrica, "init": init, "abreviatura": abrev,
                                  "n_sentidos": grupo["sentido"].nunique(), "dims": X.shape[1], **resultado})
                pd.DataFrame(filas[n_antes:]).round(4).to_csv(parcial, mode="a", index=False,
                                                              header=not os.path.exists(parcial))
                print(f"  {nombre:<15} {metrica:<9} {init:<9} {time.time() - inicio:.1f} s", flush=True)

    por_abrev = pd.DataFrame(filas)
    por_abrev.round(4).to_csv(f"{salida}/comparativa_kmeans_propio_por_abreviatura.csv", index=False)
    comparativa = (por_abrev.groupby(["metodo", "metrica", "init"], sort=False)[["dims", "ARI", "NMI", "Pureza", "ARI_medio", "ARI_std"]]
                   .mean().round(4).sort_values("ARI", ascending=False))

    comparativa.to_csv(f"{salida}/comparativa_kmeans_propio.csv")
    os.remove(parcial)  # ya está todo en el archivo completo
    print("\nComparativa con nuestro K-Means (media sobre abreviaturas; k = nº de sentidos; "
          "dims = dimensiones medias):")
    print(comparativa.to_string())
    metodo, metrica, init = comparativa.index[0]
    print(f"\nMejor representación: {metodo} ({metrica}, {init})")
    print(f"Resultados guardados en '{salida}/'")


if __name__ == "__main__":
    sys.exit(main())
