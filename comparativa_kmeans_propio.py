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

Con --todos-juntos se compara igual que agrupa el modelo final (train.py): los textos de todas las abreviaturas
mezclados en un único K-Means, con la misma k para todas las representaciones (cada una de --k). Para que sea
rápido se usan --n-abreviaturas al azar, cada una con TODOS sus textos. Con k mucho menor que el nº de sentidos, el
ARI global no dice nada, así que se ordena por el ARI dentro de cada abreviatura: se agrupa todo junto y luego,
para cada abreviatura, se compara el cluster de sus textos con su sentido (¿el modelo global separa los sentidos
de "AD"?), y se hace la media. También se dan el NMI frente a sentidos y frente a abreviaturas. Las dispersas
(BoW, TF-IDF) se quedan con las --max-palabras columnas más frecuentes: con todo el vocabulario no caben en memoria.
    python comparativa_kmeans_propio.py --todos-juntos --varianza-pca 0.8
Resultados en <representaciones>/todos_juntos/ (o en --salida): comparativa_todos_juntos.csv y comparativa_todos_juntos_por_k.csv
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse  # solo para leer las matrices dispersas (.npz) que guarda representacion_vectorial.py

from evaluacion import ari, evaluar_con_etiquetas, nmi, pureza
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


def ari_por_abreviatura(abreviaturas, sentidos, etiquetas):
    """Media sobre las abreviaturas del ARI entre sus sentidos y los clusters del modelo global en que caen sus textos."""
    datos = pd.DataFrame({"abreviatura": abreviaturas, "sentido": sentidos, "cluster": etiquetas})
    # Con una muestra de textos alguna abreviatura puede quedarse con 1 solo texto: ahí el ARI no está definido
    return float(np.mean([ari(g["sentido"].to_numpy(), g["cluster"].to_numpy())
                          for _, g in datos.groupby("abreviatura") if len(g) > 1]))


def comparativa_todos_juntos(df, matrices, variantes, metricas, inits, semillas, ks, n_abreviaturas, max_palabras,
                             salida):
    """Agrupa los textos de muchas abreviaturas juntos con la misma k en cada representación (como train.py)."""
    elegidas = sorted(df["abreviatura"].unique())
    if 0 < n_abreviaturas < len(elegidas):
        elegidas = np.random.default_rng(semillas[0]).choice(elegidas, n_abreviaturas, replace=False)
    df = df[df["abreviatura"].isin(elegidas)]
    print(f"Todos juntos: {len(df)} textos de {df['abreviatura'].nunique()} abreviaturas y "
          f"{df['sentido'].nunique()} sentidos; k = {', '.join(map(str, ks))}")
    sentidos, abreviaturas = df["sentido"].to_numpy(), df["abreviatura"].to_numpy()

    filas = []
    for nombre, (origen, varianza) in variantes.items():
        X = matrices[origen][df.index.to_numpy()]
        if sparse.issparse(X):
            frecuentes = np.argsort(-X.getnnz(axis=0))[:max_palabras]
            X = X[:, np.sort(frecuentes)].toarray()
        X = np.asarray(X, dtype=np.float64)
        # Algún texto puede quedarse sin ninguna de las palabras frecuentes: vector vacío, se cuenta en su propio
        # grupo para no romper el coseno (le damos una dimensión extra que solo usa él)
        vacios = ~np.any(X, axis=1)
        if vacios.any():
            X = np.hstack([X, vacios[:, None].astype(np.float64)])
        if varianza:
            X = pca(X, varianza)
        for metrica in metricas:
            for init in inits:
                if init == "kmeans||" and metrica != "euclidea":
                    continue
                inicio = time.time()
                for k in ks:
                    mejor = None
                    for semilla in semillas:
                        km = KMeans(p_cluster_number=k, p_metric=METRICAS[metrica](), p_init=INICIALIZACIONES[init]())
                        km.fit(X, semilla)
                        resultado = (inercia(km, X), km.predict(X))
                        if mejor is None or resultado[0] < mejor[0]:
                            mejor = resultado
                    etiquetas = mejor[1]
                    filas.append({"metodo": nombre, "metrica": metrica, "init": init, "k": k, "dims": X.shape[1],
                                  "ARI_por_abreviatura": ari_por_abreviatura(abreviaturas, sentidos, etiquetas),
                                  "NMI_sentidos": nmi(sentidos, etiquetas),
                                  "NMI_abreviaturas": nmi(abreviaturas, etiquetas),
                                  "ARI_sentidos": ari(sentidos, etiquetas),
                                  "Pureza_sentidos": pureza(sentidos, etiquetas)})
                print(f"  {nombre:<15} {metrica:<9} {init:<9} {time.time() - inicio:.1f} s", flush=True)

    os.makedirs(salida, exist_ok=True)
    por_k = pd.DataFrame(filas)
    por_k.round(4).to_csv(f"{salida}/comparativa_todos_juntos_por_k.csv", index=False)
    comparativa = (por_k.groupby(["metodo", "metrica", "init"], sort=False)
                   [["dims", "ARI_por_abreviatura", "NMI_sentidos", "NMI_abreviaturas", "ARI_sentidos",
                     "Pureza_sentidos"]].mean().round(4).sort_values("ARI_por_abreviatura", ascending=False))
    comparativa.to_csv(f"{salida}/comparativa_todos_juntos.csv")
    print(f"\nComparativa con todos los textos juntos (media sobre k = {', '.join(map(str, ks))}):")
    print(comparativa.to_string())
    metodo, metrica, init = comparativa.index[0]
    print(f"\nMejor representación: {metodo} ({metrica}, {init})")
    print(f"Resultados guardados en '{salida}/'")


def main():
    parser = argparse.ArgumentParser(description="Comparativa de representaciones con nuestro K-Means")
    parser.add_argument("--representaciones", default="representaciones",
                        help="Carpeta con instancias.csv y X_<metodo>.npy/.npz")
    parser.add_argument("--metodos", nargs="+", default=None, help="Por defecto, todas las X_* que haya")
    parser.add_argument("--metrica", nargs="+", default=["coseno"], choices=list(METRICAS))
    parser.add_argument("--init", nargs="+", default=["kmeans++"], choices=list(INICIALIZACIONES))
    parser.add_argument("--n-semillas", type=int, default=10)
    parser.add_argument("--varianza-pca", nargs="+", type=float, default=[0.9],
                        help="Fracción de varianza que deben explicar las variantes '+pca' (0 = no calcularlas)")
    parser.add_argument("--salida", default=None, help="Por defecto, la carpeta de --representaciones")
    parser.add_argument("--todos-juntos", action="store_true",
                        help="Agrupar todos los textos juntos con la misma k (como train.py), no por abreviatura")
    parser.add_argument("--n-abreviaturas", type=int, default=150,
                        help="Con --todos-juntos: abreviaturas al azar, con todos sus textos (0 = todas)")
    parser.add_argument("--k", nargs="+", type=int, default=[20, 50, 100], help="Con --todos-juntos: k a probar")
    parser.add_argument("--max-palabras", type=int, default=5000,
                        help="Con --todos-juntos: columnas más frecuentes que se quedan las dispersas")
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
    if args.todos_juntos:
        # Por defecto en su propia carpeta, para no mezclarse con los resultados por abreviatura
        salida = args.salida or f"{args.representaciones}/todos_juntos"
        return comparativa_todos_juntos(df, matrices, variantes, args.metrica, args.init, semillas, args.k,
                                        args.n_abreviaturas, args.max_palabras, salida)

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
