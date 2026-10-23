"""
Búsqueda dicotómica del número de clusters k, con nuestro K-Means y la silueta de evaluacion.py.
Versión "todos juntos": un único modelo para todos los textos. La versión con un modelo por abreviatura (una k
para cada una) es buscar_k.py.

Se agrupan TODOS los textos de MeDAL juntos (no una abreviatura cada vez), así que hay que elegir una única k.
No hay una k "real" con la que compararla: el conjunto tiene miles de sentidos y de abreviaturas, así que la k se
elige sin mirar las etiquetas. La silueta (métrica interna) dice lo compactos y separados que son los clusters:
se busca la k que la maximiza. En vez de probar todas las k del rango (búsqueda exhaustiva), la búsqueda
dicotómica parte el rango por la mitad en cada paso:

    mientras el rango [k_min, k_max] tenga más de un valor:
        m = punto medio
        si silueta(m) < silueta(m + 1): el máximo está a la derecha  -> k_min = m + 1
        si no:                          el máximo está a la izquierda -> k_max = m

Con un rango de n valores hace unas 2·log2(n) ejecuciones en vez de n. Solo garantiza el mejor k si la silueta
sube hasta un máximo y luego baja (función unimodal); si tiene varios picos puede quedarse en uno local.

Si la silueta tiene el máximo en k pequeñas y luego una cola plana con ruido, la dicotómica empieza en el centro
del rango, cae en esa cola y una subida minúscula la aleja del pico. Por eso hay una segunda versión,
busqueda_exponencial_dicotomica: primero localiza la montaña probando k = 2, 4, 8, 16... (doblando) mientras la
silueta sube, y luego hace la dicotómica solo en ese tramo.

Una vez elegida la k, se mira si sirve para lo que queremos: el ARI dentro de cada abreviatura (¿los textos de
"AD" con sentidos distintos caen en clusters distintos?), igual que en comparativa_kmeans_propio.py --todos-juntos.

La búsqueda exhaustiva se usa de referencia: dice cuál es de verdad la k con mejor silueta del rango, y así se
ve si las dos dicotómicas la encuentran y cuántos entrenamientos ahorran. Entrenar el K-Means muchas veces con
los 257.500 textos es lento, así que la búsqueda se hace sobre una muestra al azar (--n-textos).

Uso (después de comparativa_kmeans_propio.py, que decide la representación):
    python buscar_k_todos.py --n-textos 0 --sin-exhaustiva   # todos los textos (la exhaustiva tardaría horas)
    python buscar_k_todos.py --metodo w2v --n-textos 50000 --k-max 200
    python buscar_k_todos.py --n-textos 0 --k-min 10 --k-max 100 --sin-exhaustiva --salida <carpeta>   # rango acotado

Sin acotar, con todos los textos juntos la silueta es casi plana y su máximo cae en k = 2 (dos grupos para 257.500
textos no separan nada útil); con --k-min se busca solo a partir de una k mínima razonable.

Para usarla en el modelo final (p. ej. desde find_cluster_number de train.py):
    from buscar_k_todos import busqueda_exponencial_dicotomica, silueta_con_k
    k, _ = busqueda_exponencial_dicotomica(lambda k: silueta_con_k(X, k), 2, 100)

Resultados en <representaciones>/todos_juntos/buscar_k_resumen.csv (la k de cada búsqueda, entrenamientos y segundos) y
<representaciones>/todos_juntos/buscar_k_siluetas.csv (silueta de cada k evaluada, para dibujar la curva)
"""

import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse

from comparativa_kmeans_propio import INICIALIZACIONES, METRICAS, ari_por_abreviatura, cargar_matrices, inercia, pca
from evaluacion import ari, evaluar_clustering, nmi
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
    parser.add_argument("--k-min", type=int, default=2, help="Menor k que se busca")
    parser.add_argument("--k-max", type=int, default=100, help="Mayor k que se busca")
    parser.add_argument("--salida", default=None, help="Carpeta de resultados (por defecto, <representaciones>/todos_juntos)")
    parser.add_argument("--n-textos", type=int, default=20000, help="Textos al azar sobre los que se busca (0 = todos)")
    parser.add_argument("--n-semillas", type=int, default=1)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-exhaustiva", action="store_true", help="No hacer la búsqueda exhaustiva (la más lenta)")
    args = parser.parse_args()

    metodo = args.metodo or mejor_metodo(args.representaciones)
    origen, _, porcentaje = re.match(r"([^+]+)(\+pca(\d+)%)?$", metodo).groups()
    varianza = int(porcentaje) / 100 if porcentaje else None

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido"])
    M = cargar_matrices(args.representaciones, [origen])[origen]
    # Igual que en la comparativa: fuera las instancias con vector vacío
    df = df[(M.getnnz(axis=1) > 0) if sparse.issparse(M) else np.any(M, axis=1)]
    if 0 < args.n_textos < len(df):
        df = df.loc[np.sort(np.random.default_rng(args.semilla).choice(df.index, args.n_textos, replace=False))]
    X = M[df.index.to_numpy()]
    if sparse.issparse(X):
        X = X[:, X.getnnz(axis=0) > 0].toarray()
    X = np.asarray(X, dtype=np.float64)
    if varianza:
        X = pca(X, varianza)
    k_max = min(args.k_max, len(X) - 1)
    semillas = [args.semilla + i for i in range(args.n_semillas)]
    print(f"Representación: {metodo} ({args.metrica}, {args.init}); {len(X)} textos de "
          f"{df['abreviatura'].nunique()} abreviaturas y {df['sentido'].nunique()} sentidos; "
          f"{X.shape[1]} dimensiones; k entre {args.k_min} y {k_max}", flush=True)

    # Las tres búsquedas consultan la misma función (mismas semillas), así que cada K-Means se entrena una sola
    # vez y todas ven las mismas siluetas. Por eso el tiempo de cada búsqueda no se cronometra directamente (las
    # que van después saldrían gratis): se mide lo que tarda cada k una vez y a cada búsqueda se le suman las k
    # que ha consultado
    siluetas, etiquetas, segundos = {}, {}, {}

    def evaluar(k):
        if k not in siluetas:
            inicio = time.perf_counter()
            siluetas[k] = silueta_con_k(X, k, args.metrica, args.init, semillas, etiquetas)
            segundos[k] = time.perf_counter() - inicio
            print(f"  k = {k}: silueta {siluetas[k]:.4f} ({segundos[k]:.1f} s)", flush=True)
        return siluetas[k]

    def anotando(consultadas):
        return lambda k: (consultadas.add(k), evaluar(k))[1]

    busquedas = [("dicotómica", busqueda_dicotomica), ("exponencial + dicotómica", busqueda_exponencial_dicotomica)]
    if not args.sin_exhaustiva:
        busquedas.append(("exhaustiva", busqueda_exhaustiva))
    filas = []
    for nombre, buscar in busquedas:
        print(f"Búsqueda {nombre}:", flush=True)
        consultadas = set()
        k, n = buscar(anotando(consultadas), args.k_min, k_max)
        filas.append({"busqueda": nombre, "k": k, "silueta": siluetas[k], "entrenamientos": n,
                      "segundos": sum(segundos[c] for c in consultadas),
                      # ¿Separa el modelo global los sentidos de cada abreviatura? (como la comparativa --todos-juntos)
                      "ARI_por_abreviatura": ari_por_abreviatura(df["abreviatura"].to_numpy(),
                                                                 df["sentido"].to_numpy(), etiquetas[k]),
                      # Solo orientativo: con k mucho menor que el nº de sentidos, el ARI global tiene que salir bajo
                      "ARI_sentidos": ari(df["sentido"].to_numpy(), etiquetas[k]),
                      "NMI_sentidos": nmi(df["sentido"].to_numpy(), etiquetas[k]),
                      "NMI_abreviaturas": nmi(df["abreviatura"].to_numpy(), etiquetas[k])})

    salida = args.salida or f"{args.representaciones}/todos_juntos"
    os.makedirs(salida, exist_ok=True)
    r = pd.DataFrame(filas).set_index("busqueda")
    r.round(4).to_csv(f"{salida}/buscar_k_resumen.csv")
    pd.Series(siluetas, name="silueta").sort_index().rename_axis("k").round(4).to_csv(f"{salida}/buscar_k_siluetas.csv")
    print(f"\nResultados ({len(X)} textos, k entre {args.k_min} y {k_max}):")
    print(r.round(3).to_string())
    print(f"Detalle en '{salida}/buscar_k_resumen.csv' y '{salida}/buscar_k_siluetas.csv'")


if __name__ == "__main__":
    main()
