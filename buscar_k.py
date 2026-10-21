"""
Búsqueda dicotómica del número de clusters k, con nuestro K-Means y la silueta de evaluacion.py.

En la vida real no se sabe cuántos sentidos tiene una abreviatura, así que hay que elegir k sin mirar las
etiquetas. La silueta (métrica interna) dice lo compactos y separados que son los clusters: se busca la k que
la maximiza. En vez de probar todas las k del rango (búsqueda exhaustiva), la búsqueda dicotómica parte el
rango por la mitad en cada paso:

    mientras el rango [k_min, k_max] tenga más de un valor:
        m = punto medio
        si silueta(m) < silueta(m + 1): el máximo está a la derecha  -> k_min = m + 1
        si no:                          el máximo está a la izquierda -> k_max = m

Con un rango de n valores hace unas 2·log2(n) ejecuciones en vez de n. Solo garantiza el mejor k si la silueta
sube hasta un máximo y luego baja (función unimodal); si tiene varios picos puede quedarse en uno local.

En MeDAL la silueta suele tener el máximo en k pequeñas y luego una cola plana con ruido: la dicotómica empieza en
el centro del rango, cae en esa cola y una subida minúscula la aleja del pico. Por eso hay una segunda versión,
busqueda_exponencial_dicotomica: primero localiza la montaña probando k = 2, 4, 8, 16... (doblando) mientras la
silueta sube, y luego hace la dicotómica solo en ese tramo.

Como MeDAL trae los sentidos, el experimento de este archivo comprueba cuánto acierta: compara la k que encuentra
la búsqueda dicotómica con la k real (nº de sentidos) y con la que encuentra la búsqueda exhaustiva.

Uso (después de comparativa_kmeans_propio.py, que decide la representación):
    python buscar_k.py                                   # la mejor representación de la comparativa
    python buscar_k.py --metodo w2v --n-abreviaturas 500 --k-max 20

Para usarla en el modelo final (p. ej. desde find_cluster_number de train.py):
    from buscar_k import busqueda_dicotomica, silueta_con_k
    k, _ = busqueda_exponencial_dicotomica(lambda k: silueta_con_k(X, k), 2, 20)

Resultados en <representaciones>/buscar_k_por_abreviatura.csv
"""

import argparse
import contextlib
import io
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse

from comparativa_kmeans_propio import INICIALIZACIONES, METRICAS, cargar_matrices, inercia, pca
from evaluacion import ari, evaluar_clustering
from modeloKMeans import KMeans


def mejor_ejecucion(X, k, metrica, init, semillas):
    """Etiquetas de la ejecución de nuestro K-Means con menor inercia entre las semillas (sin mirar sentidos)."""
    mejor = None
    for semilla in semillas:
        km = KMeans(p_cluster_number=k, p_metric=METRICAS[metrica](), p_init=INICIALIZACIONES[init]())
        km.fit(X, semilla)
        resultado = (inercia(km, X), km.predict(X))
        if mejor is None or resultado[0] < mejor[0]:
            mejor = resultado
    return mejor[1]


def silueta_con_k(X, k, metrica="coseno", init="kmeans++", semillas=(42,), etiquetas_por_k=None):
    """Silueta (evaluacion.py) del clustering de X en k grupos; -1 si el K-Means deja menos de 2 clusters."""
    etiquetas = mejor_ejecucion(X, k, metrica, init, semillas)
    if etiquetas_por_k is not None:
        etiquetas_por_k[k] = etiquetas
    # Con coseno la silueta se mide sobre los vectores normalizados, que son los que agrupa el K-Means;
    # evaluar_clustering imprime un resumen en cada llamada y aquí se llama muchas veces, así que se silencia
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluar_clustering(METRICAS[metrica].format_data(X), etiquetas)["silhouette_score"]
    except ValueError:
        return -1.0


def con_memoria(evaluar):
    """evaluar(k) que no repite cálculos; .valores guarda las k evaluadas (para contar entrenamientos)."""
    valores = {}

    def f(k):
        if k not in valores:
            valores[k] = evaluar(k)
        return valores[k]
    f.valores = valores
    return f


def busqueda_dicotomica(evaluar, k_min, k_max):
    """
    Busca la k de [k_min, k_max] que maximiza evaluar(k) partiendo el rango por la mitad en cada paso.
    Devuelve (k, nº de valores de k evaluados). Cada k se evalúa como mucho una vez.
    """
    f = evaluar if hasattr(evaluar, "valores") else con_memoria(evaluar)
    while k_min < k_max:
        m = (k_min + k_max) // 2
        if f(m) < f(m + 1):
            k_min = m + 1
        else:
            k_max = m
    return k_min, len(f.valores)


def busqueda_exponencial_dicotomica(evaluar, k_min, k_max):
    """
    Primero dobla k (k_min, 2·k_min, 4·k_min...) mientras evaluar(k) mejore, para localizar el tramo donde está
    el máximo; luego hace la búsqueda dicotómica solo en ese tramo. Devuelve (k, nº de valores de k evaluados).
    """
    f = con_memoria(evaluar)
    anterior, actual = k_min, k_min
    siguiente = min(2 * actual, k_max)
    while siguiente > actual and f(siguiente) > f(actual):
        anterior, actual = actual, siguiente
        siguiente = min(2 * actual, k_max)
    # El máximo está entre la k de antes de la última subida y la primera k en la que ya no mejora
    return busqueda_dicotomica(f, anterior, siguiente)


def busqueda_exhaustiva(evaluar, k_min, k_max):
    """Prueba todas las k de [k_min, k_max] y devuelve (la de mayor evaluar(k), nº de valores evaluados)."""
    valores = {k: evaluar(k) for k in range(k_min, k_max + 1)}
    return max(valores, key=valores.get), len(valores)


def mejor_metodo(directorio):
    ruta = f"{directorio}/comparativa_kmeans_propio.csv"
    if not os.path.exists(ruta):
        raise SystemExit("No hay comparativa_kmeans_propio.csv: ejecuta antes la comparativa o usa --metodo")
    return pd.read_csv(ruta)["metodo"].iloc[0]


def main():
    parser = argparse.ArgumentParser(description="Búsqueda dicotómica de k con nuestro K-Means")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--metodo", default=None,
                        help="Representación (p. ej. w2v o w2v+pca90%%); por defecto, la mejor de la comparativa")
    parser.add_argument("--metrica", default="coseno", choices=list(METRICAS))
    parser.add_argument("--init", default="kmeans++", choices=list(INICIALIZACIONES))
    parser.add_argument("--k-max", type=int, default=20, help="Mayor k que se busca (como mucho, nº de contextos - 1)")
    parser.add_argument("--n-abreviaturas", type=int, default=300, help="Abreviaturas al azar (0 = todas)")
    parser.add_argument("--n-semillas", type=int, default=3)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    metodo = args.metodo or mejor_metodo(args.representaciones)
    origen, _, porcentaje = re.match(r"([^+]+)(\+pca(\d+)%)?$", metodo).groups()
    varianza = int(porcentaje) / 100 if porcentaje else None

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido"])
    M = cargar_matrices(args.representaciones, [origen])[origen]
    # Igual que en la comparativa: fuera las instancias con vector vacío
    df = df[(M.getnnz(axis=1) > 0) if sparse.issparse(M) else np.any(M, axis=1)]
    abreviaturas = sorted(df["abreviatura"].unique())
    if 0 < args.n_abreviaturas < len(abreviaturas):
        abreviaturas = sorted(np.random.default_rng(args.semilla).choice(abreviaturas, args.n_abreviaturas,
                                                                         replace=False))
    semillas = [args.semilla + i for i in range(args.n_semillas)]
    print(f"Representación: {metodo} ({args.metrica}, {args.init}); {len(abreviaturas)} abreviaturas; "
          f"k entre 2 y {args.k_max}")

    filas = []
    for n, (abrev, grupo) in enumerate(df[df["abreviatura"].isin(abreviaturas)].groupby("abreviatura"), 1):
        X = M[grupo.index.to_numpy()]
        if sparse.issparse(X):
            X = X[:, X.getnnz(axis=0) > 0].toarray()
        X = np.asarray(X, dtype=np.float64)
        if varianza:
            X = pca(X, varianza)
        sentidos = grupo["sentido"].to_numpy()
        k_real = len(np.unique(sentidos))
        k_max = min(args.k_max, len(X) - 1)
        if k_max < 2:
            continue

        # La exhaustiva calcula la silueta de todas las k; la dicotómica consulta las mismas (mismas semillas),
        # así que dan exactamente lo mismo que por separado y cada K-Means se entrena una sola vez
        siluetas, etiquetas = {}, {}
        evaluar = lambda k: siluetas.setdefault(k, silueta_con_k(X, k, args.metrica, args.init, semillas, etiquetas))
        k_exh, n_exh = busqueda_exhaustiva(evaluar, 2, k_max)
        k_dic, n_dic = busqueda_dicotomica(evaluar, 2, k_max)
        k_exp, n_exp = busqueda_exponencial_dicotomica(evaluar, 2, k_max)
        filas.append({"abreviatura": abrev, "n_contextos": len(X), "k_real": k_real,
                      "k_dicotomica": k_dic, "k_exp_dicotomica": k_exp, "k_exhaustiva": k_exh,
                      "evaluaciones_dicotomica": n_dic, "evaluaciones_exp_dicotomica": n_exp,
                      "evaluaciones_exhaustiva": n_exh,
                      "ARI_dicotomica": ari(sentidos, etiquetas[k_dic]),
                      "ARI_exp_dicotomica": ari(sentidos, etiquetas[k_exp]),
                      "ARI_exhaustiva": ari(sentidos, etiquetas[k_exh]),
                      "ARI_k_real": ari(sentidos, etiquetas[k_real]) if k_real <= k_max else np.nan})
        if n % 50 == 0:
            print(f"  {n}/{len(abreviaturas)} abreviaturas", flush=True)

    r = pd.DataFrame(filas)
    r.round(4).to_csv(f"{args.representaciones}/buscar_k_por_abreviatura.csv", index=False)
    print(f"\nResultados sobre {len(r)} abreviaturas (k real: mediana {r['k_real'].median():.0f}, "
          f"máx. {r['k_real'].max()}; en {(r['k_real'] > args.k_max).sum()} la k real queda fuera del rango):")
    resumen = pd.DataFrame({
        nombre: {"misma k que la exhaustiva": (r[f"k_{col}"] == r["k_exhaustiva"]).mean(),
                 "acierta la k real": (r[f"k_{col}"] == r["k_real"]).mean(),
                 "error medio |k - k real|": (r[f"k_{col}"] - r["k_real"]).abs().mean(),
                 "entrenamientos por abreviatura": r[f"evaluaciones_{col}"].mean(),
                 "ARI medio": r[f"ARI_{col}"].mean()}
        for nombre, col in [("dicotómica", "dicotomica"), ("exponencial + dicotómica", "exp_dicotomica"),
                            ("exhaustiva", "exhaustiva")]})
    print(resumen.round(3).to_string())
    print(f"(ARI medio usando la k real: {r['ARI_k_real'].mean():.3f})")
    print(f"Detalle en '{args.representaciones}/buscar_k_por_abreviatura.csv'")


if __name__ == "__main__":
    main()
