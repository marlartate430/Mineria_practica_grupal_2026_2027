"""
Modelo unificado: un único modelo que guarda dentro un K-Means por abreviatura.

Cada abreviatura tiene su propio PCA y sus propios centroides (los de NUESTRO K-Means, modeloKMeans.py), pero todo
se guarda en un solo archivo y se usa con una sola función:

    modelo = ModeloUnificado.cargar("modelos_unificado/modelo_unificado.npz")
    clusters = modelo.asignar(vectores, abreviaturas)     # el cluster de cada texto dentro de SU abreviatura

Es como un único K-Means con todos los textos en el que un texto de "AD" solo puede caer en un cluster de "AD"
(como si la distancia entre abreviaturas distintas fuese infinita). El número total de clusters es la suma de las
k de todas las abreviaturas.

La k de cada abreviatura se elige SIN mirar los sentidos: la que maximiza la silueta, con la búsqueda exponencial +
dicotómica de buscar_k.py (k entre 2 y --k-max). Para comprobar que funciona con textos nuevos, en cada abreviatura
se aparta un --test de sus textos (estratificado por sentido): el PCA y el K-Means se ajustan solo con el resto, y
los apartados se asignan al centroide más cercano con asignar(). La calidad se mide con evaluacion.py: índices
internos (silueta, Calinski-Harabasz, Davies-Bouldin; no usan los sentidos) e índices externos. Los sentidos solo se
usan al final, para medir
ARI, NMI y pureza en entrenamiento y en prueba.

Uso (por defecto, la combinación ganadora de comparativa_kmeans_propio.csv: representación, PCA, métrica e init):
    python modelo_unificado.py
    python modelo_unificado.py --metodo st --varianza-pca 0.8 --metrica coseno --init kmeans++ --n-abreviaturas 300

Resultados en modelos_unificado/:
    modelo_unificado.npz             el modelo (PCA y centroides de cada abreviatura)
    modelo_unificado_resumen.json    configuración y métricas medias
    modelo_unificado_por_abreviatura.csv   k elegida, k real y métricas de cada abreviatura
"""

import argparse
import contextlib
import io
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse

from buscar_k import busqueda_exponencial_dicotomica, silueta_con_k
from comparativa_kmeans_propio import INICIALIZACIONES, METRICAS, cargar_matrices, inercia
from evaluacion import evaluar_clustering, evaluar_con_etiquetas
from modeloKMeans import KMeans


def ajustar_pca(X, varianza):
    """
    El mismo PCA que comparativa_kmeans_propio.pca, pero devolviendo la media y las componentes para poder
    transformar después textos nuevos igual: Z = (X - media) · componentesᵀ.
    """
    media = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - media, full_matrices=False)
    acumulada = np.cumsum(S ** 2) / np.sum(S ** 2)
    n_componentes = int(np.searchsorted(acumulada, varianza) + 1)
    return media, Vt[:n_componentes]


class ModeloUnificado:
    def __init__(self, metrica="coseno", init="random", varianza_pca=0.8, k_max=20, semillas=(42, 43, 44),
                 metodo="bert_media"):
        self.metodo = metodo  # la representación (matriz X_<metodo>) con la que se entrena
        self.metrica = metrica
        self.init = init
        self.varianza_pca = varianza_pca
        self.k_max = k_max
        self.semillas = list(semillas)
        # abreviatura -> {"media", "componentes", "centroides"} (media y componentes son None sin PCA)
        self.partes = {}

    def _transformar(self, abrev, X):
        parte = self.partes[abrev]
        return X if parte["media"] is None else (X - parte["media"]) @ parte["componentes"].T

    def fit_abreviatura(self, abrev, X):
        """Ajusta el PCA y el K-Means de una abreviatura; la k es la de mayor silueta (no mira los sentidos)."""
        media, componentes = ajustar_pca(X, self.varianza_pca) if self.varianza_pca else (None, None)
        self.partes[abrev] = {"media": media, "componentes": componentes, "centroides": None}
        Z = self._transformar(abrev, X)
        k_max = min(self.k_max, len(Z) - 1)
        k, _ = busqueda_exponencial_dicotomica(
            lambda k: silueta_con_k(Z, k, self.metrica, self.init, self.semillas), 2, k_max)
        # Con la k elegida, la ejecución de menor inercia entre las semillas (sin mirar los sentidos)
        mejor = None
        for semilla in self.semillas:
            km = KMeans(p_cluster_number=k, p_metric=METRICAS[self.metrica](), p_init=INICIALIZACIONES[self.init]())
            km.fit(Z, semilla)
            if mejor is None or inercia(km, Z) < mejor[0]:
                mejor = (inercia(km, Z), km)
        self.partes[abrev]["centroides"] = mejor[1].centroids
        return k

    def asignar(self, X, abreviaturas):
        """Asigna cada fila de X al cluster del centroide más cercano, solo entre los de SU abreviatura."""
        abreviaturas = np.asarray(abreviaturas)
        etiquetas = np.empty(len(X), dtype=int)
        for abrev in np.unique(abreviaturas):
            filas = abreviaturas == abrev
            Z = self._transformar(abrev, X[filas])
            metrica = METRICAS[self.metrica]()
            etiquetas[filas] = np.argmin(metrica.get_distance(metrica.format_data(Z),
                                                              self.partes[abrev]["centroides"]), axis=1)
        return etiquetas

    def n_clusters(self):
        return sum(len(p["centroides"]) for p in self.partes.values())

    def guardar(self, ruta):
        # Un .npz con tres arrays por abreviatura; en float32 para que ocupe la mitad
        arrays = {"config": np.array(json.dumps({"metodo": self.metodo, "metrica": self.metrica, "init": self.init,
                                                 "varianza_pca": self.varianza_pca, "k_max": self.k_max,
                                                 "semillas": self.semillas}))}
        for i, (abrev, parte) in enumerate(self.partes.items()):
            arrays[f"{i}_abreviatura"] = np.array(abrev)
            arrays[f"{i}_centroides"] = parte["centroides"].astype(np.float32)
            if parte["media"] is not None:
                arrays[f"{i}_media"] = parte["media"].astype(np.float32)
                arrays[f"{i}_componentes"] = parte["componentes"].astype(np.float32)
        np.savez_compressed(ruta, **arrays)

    @classmethod
    def cargar(cls, ruta):
        datos = np.load(ruta)
        modelo = cls(**json.loads(str(datos["config"])))
        i = 0
        while f"{i}_abreviatura" in datos:
            modelo.partes[str(datos[f"{i}_abreviatura"])] = {
                "media": datos[f"{i}_media"].astype(np.float64) if f"{i}_media" in datos else None,
                "componentes": datos[f"{i}_componentes"].astype(np.float64) if f"{i}_componentes" in datos else None,
                "centroides": datos[f"{i}_centroides"].astype(np.float64)}
            i += 1
        return modelo


INTERNAS = ("silueta", "calinski_harabasz", "davies_bouldin")
EXTERNAS = ("ARI", "NMI", "Pureza")
METRICAS_RESUMEN = INTERNAS + EXTERNAS


def indices_internos(modelo, abrev, X, etiquetas):
    """
    Silueta, Calinski-Harabasz y Davies-Bouldin (evaluar_clustering de evaluacion.py) en el espacio en el que agrupa
    el K-Means (PCA de la abreviatura y, con coseno, vectores normalizados). NaN si hay menos de 2 clusters.
    """
    Z = METRICAS[modelo.metrica].format_data(modelo._transformar(abrev, X))
    try:
        # evaluar_clustering imprime un resumen en cada llamada y aquí se llama miles de veces, así que se silencia
        with contextlib.redirect_stdout(io.StringIO()):
            r = evaluar_clustering(Z, etiquetas)
    except ValueError:
        return {m: np.nan for m in INTERNAS}
    return {"silueta": r["silhouette_score"], "calinski_harabasz": r["calinski_harabasz_score"],
            "davies_bouldin": r["davies_bouldin_score"]}


def vectores(M, indices):
    """Filas de una matriz de representaciones (densa .npy o dispersa .npz) como array denso de float64."""
    X = M[indices]
    return np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float64)


def ganadora(directorio):
    """(metodo, varianza del PCA o None, metrica, init) de la primera fila de comparativa_kmeans_propio.csv."""
    ruta = f"{directorio}/comparativa_kmeans_propio.csv"
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe '{ruta}': ejecuta antes comparativa_kmeans_propio.py o indica --metodo, "
                         "--varianza-pca, --metrica e --init")
    mejor = pd.read_csv(ruta).iloc[0]
    # 'bert_media+pca80%' -> ('bert_media', 0.8); 'w2v' -> ('w2v', None)
    metodo, _, porcentaje = re.match(r"([^+]+)(\+pca(\d+)%)?$", mejor["metodo"]).groups()
    return metodo, int(porcentaje) / 100 if porcentaje else None, mejor["metrica"], mejor["init"]


def separar_prueba(grupo, fraccion, rng):
    """Índices de prueba: una fracción de los textos de cada sentido (al menos 1 si el sentido tiene 2 o más)."""
    prueba = []
    for _, textos in grupo.groupby("sentido"):
        n = int(round(len(textos) * fraccion)) if len(textos) > 1 else 0
        prueba.extend(rng.choice(textos.index, max(n, 1 if len(textos) > 1 else 0), replace=False))
    return grupo.index.isin(prueba)


def main():
    parser = argparse.ArgumentParser(description="Modelo unificado: un K-Means por abreviatura en un solo modelo")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--metodo", default=None,
                        help="Matriz X_<metodo> (sin el sufijo +pca); por defecto, la ganadora de la comparativa")
    parser.add_argument("--varianza-pca", type=float, default=None, help="0 = sin PCA")
    parser.add_argument("--metrica", default=None, choices=list(METRICAS))
    parser.add_argument("--init", default=None, choices=list(INICIALIZACIONES))
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--n-semillas", type=int, default=3)
    parser.add_argument("--test", type=float, default=0.2, help="Fracción de textos de prueba por sentido")
    parser.add_argument("--n-abreviaturas", type=int, default=0, help="Abreviaturas al azar (0 = todas)")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", default="modelos_unificado")
    args = parser.parse_args()
    # Lo que no se indique se toma de la combinación ganadora de la comparativa por abreviatura
    if None in (args.metodo, args.varianza_pca, args.metrica, args.init):
        metodo, varianza, metrica, init = ganadora(args.representaciones)
        args.metodo = args.metodo or metodo
        args.varianza_pca = varianza if args.varianza_pca is None else args.varianza_pca
        args.metrica, args.init = args.metrica or metrica, args.init or init

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido"])
    M = cargar_matrices(args.representaciones, [args.metodo])[args.metodo]
    # Igual que en la comparativa: fuera las instancias con vector vacío
    df = df[(M.getnnz(axis=1) > 0) if sparse.issparse(M) else np.any(M, axis=1)]
    abreviaturas = sorted(df["abreviatura"].unique())
    rng = np.random.default_rng(args.semilla)
    if 0 < args.n_abreviaturas < len(abreviaturas):
        abreviaturas = sorted(rng.choice(abreviaturas, args.n_abreviaturas, replace=False))
    df = df[df["abreviatura"].isin(abreviaturas)]
    print(f"Modelo unificado: {args.metodo}" + (f" + PCA {args.varianza_pca:.0%}" if args.varianza_pca else "") +
          f" ({args.metrica}, {args.init}); {len(abreviaturas)} abreviaturas, {len(df)} textos; "
          f"k entre 2 y {args.k_max}; {args.test:.0%} de prueba", flush=True)

    modelo = ModeloUnificado(args.metrica, args.init, args.varianza_pca or None, args.k_max,
                             [args.semilla + i for i in range(args.n_semillas)], args.metodo)
    filas, inicio = [], time.time()
    for n, (abrev, grupo) in enumerate(df.groupby("abreviatura"), 1):
        es_prueba = separar_prueba(grupo, args.test, rng)
        entrena, prueba = grupo[~es_prueba], grupo[es_prueba]
        X = vectores(M, entrena.index.to_numpy())
        if len(X) < 3:
            continue
        k = modelo.fit_abreviatura(abrev, X)
        fila = {"abreviatura": abrev, "n_entrena": len(entrena), "n_prueba": len(prueba),
                "k_real": grupo["sentido"].nunique(), "k": k}
        for nombre, parte in [("entrena", entrena), ("prueba", prueba)]:
            if len(parte) < 2:
                continue
            Xp = vectores(M, parte.index.to_numpy())
            etiquetas = modelo.asignar(Xp, parte["abreviatura"].to_numpy())
            fila.update({f"{m}_{nombre}": v for m, v in evaluar_con_etiquetas(parte["sentido"].to_numpy(),
                                                                                 etiquetas).items()})
            fila.update({f"{m}_{nombre}": v for m, v in indices_internos(modelo, abrev, Xp, etiquetas).items()})
        filas.append(fila)
        if n % 200 == 0:
            print(f"  {n}/{len(abreviaturas)} abreviaturas ({time.time() - inicio:.0f} s)", flush=True)

    os.makedirs(args.salida, exist_ok=True)
    modelo.guardar(f"{args.salida}/modelo_unificado.npz")
    r = pd.DataFrame(filas)
    r.round(4).to_csv(f"{args.salida}/modelo_unificado_por_abreviatura.csv", index=False)
    resumen = {"representacion": args.metodo, "varianza_pca": args.varianza_pca, "metrica": args.metrica,
               "init": args.init, "k_max": args.k_max, "test": args.test,
               "n_abreviaturas": len(r), "n_clusters_total": modelo.n_clusters(),
               "n_sentidos_total": int(r["k_real"].sum()),
               "acierta_k_real": float((r["k"] == r["k_real"]).mean()),
               "error_medio_k": float((r["k"] - r["k_real"]).abs().mean()),
               "segundos": round(time.time() - inicio, 1),
               **{c: float(r[c].mean()) for c in r.columns if c.rsplit("_", 1)[0] in METRICAS_RESUMEN}}
    with open(f"{args.salida}/modelo_unificado_resumen.json", "w") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    print(f"\nModelo unificado con {resumen['n_abreviaturas']} abreviaturas y {resumen['n_clusters_total']} clusters "
          f"en total ({resumen['n_sentidos_total']} sentidos reales); {resumen['segundos']:.0f} s")
    print(f"k elegida = k real en el {resumen['acierta_k_real']:.0%} de las abreviaturas "
          f"(error medio {resumen['error_medio_k']:.2f})")
    for tipo, metricas in [("Índices internos (no usan los sentidos)", INTERNAS), ("Índices externos", EXTERNAS)]:
        print(f"\n{tipo}:")
        print(r[[c for c in r.columns if c.rsplit("_", 1)[0] in metricas]].mean().round(3).to_string())
    print(f"Resultados en '{args.salida}/'")


if __name__ == "__main__":
    main()
