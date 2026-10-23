"""
Gráficos para VER si nuestro K-Means separa bien los sentidos (comparativa_kmeans_propio.py solo da números).

Para cada abreviatura elegida dibuja una figura con una fila por representación y dos mapas 2D (PCA):
    - izquierda: los contextos coloreados por su sentido REAL
    - derecha:   los mismos puntos coloreados por el cluster que les da NUESTRO K-Means (con su ARI)
A cada cluster se le da el color del sentido con el que más coincide, así que si los dos mapas se parecen,
el K-Means ha acertado. Si existe comparativa_kmeans_propio.csv, dibuja además el ARI medio de cada método.

Uso (después de comparativa_kmeans_propio.py, o sin él eligiendo las abreviaturas a mano):
    python graficos_clusters.py
    python graficos_clusters.py --abreviaturas AA CEA --metodos bert_media+pca80% w2v tfidf
    python graficos_clusters.py --metrica euclidea --init random

Si no se dan --abreviaturas y existe comparativa_kmeans_propio_por_abreviatura.csv, se eligen la mejor, la
mediana y la peor abreviatura del mejor método (entre las de 2 a 8 sentidos, para que los colores se distingan).
Si no se dan --metodos, se dibujan el ganador de la comparativa, su versión sin PCA, w2v y tfidf.

Resultados en <representaciones>/graficos/: clusters_<abreviatura>.png y ari_por_metodo.png
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from comparativa_kmeans_propio import INICIALIZACIONES, METRICAS, cargar_matrices, inercia, pca
from evaluacion import ari, nmi, pureza
from modeloKMeans import KMeans

# Paleta categórica (8 colores distinguibles también con daltonismo); a partir del 8º sentido, gris
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS = "#b4b3ad"
TEXTO, TEXTO_SUAVE, FONDO = "#0b0b0b", "#52514e", "#fcfcfb"


def proyeccion_2d(X):
    """PCA a 2 dimensiones con numpy, solo para dibujar (el K-Means usa todas las dimensiones)."""
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


def mejor_ejecucion(X, k, metrica, init, semillas):
    """Como agrupar() de la comparativa, pero devuelve las etiquetas de la ejecución con menor inercia."""
    mejor = None
    for semilla in semillas:
        km = KMeans(p_cluster_number=k, p_metric=METRICAS[metrica](), p_init=INICIALIZACIONES[init]())
        km.fit(X, semilla)
        resultado = (inercia(km, X), km.predict(X))
        if mejor is None or resultado[0] < mejor[0]:
            mejor = resultado
    return mejor[1]


def emparejar(sentidos, etiquetas):
    """Cluster -> índice del sentido con el que más instancias comparte (para pintarlos del mismo color)."""
    tabla = pd.crosstab(etiquetas, sentidos)
    return {cluster: tabla.columns.get_loc(fila.idxmax()) for cluster, fila in tabla.iterrows()}


def pintar(ax, P, indices_color, titulo):
    colores = [COLORES[i] if i < len(COLORES) else GRIS for i in indices_color]
    ax.scatter(P[:, 0], P[:, 1], c=colores, s=10, alpha=0.7, linewidths=0)
    ax.set_title(titulo, fontsize=10, color=TEXTO, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_color("#d6d5d0")


def separar_variante(variante):
    """'bert_media+pca80%' -> ('bert_media', 0.8); 'w2v' -> ('w2v', None)."""
    origen, _, porcentaje = re.match(r"([^+]+)(\+pca(\d+)%)?$", variante).groups()
    return origen, int(porcentaje) / 100 if porcentaje else None


def metodos_por_defecto(ruta_resumen):
    """El ganador de la comparativa, su versión sin PCA y dos referencias (w2v y tfidf)."""
    if not os.path.exists(ruta_resumen):
        return None
    ganador = pd.read_csv(ruta_resumen)["metodo"].iloc[0]
    return list(dict.fromkeys([ganador, separar_variante(ganador)[0], "w2v", "tfidf"]))


def figura_abreviatura(abrev, grupo, matrices, variantes, args, semillas, salida):
    sentidos = grupo["sentido"].to_numpy()
    # Los colores van a los sentidos más frecuentes; si hay más de 8, el resto sale en gris
    nombres_sentidos = list(pd.Series(sentidos).value_counts().index)
    indice_sentido = np.array([nombres_sentidos.index(s) for s in sentidos])
    k = len(nombres_sentidos)

    fig, ejes = plt.subplots(len(variantes), 2, figsize=(10, 3.6 * len(variantes)), squeeze=False)
    fig.patch.set_facecolor(FONDO)
    for fila, metodo in zip(ejes, variantes):
        origen, varianza = separar_variante(metodo)
        X = matrices[origen][grupo.index.to_numpy()]
        if sparse.issparse(X):
            X = X[:, X.getnnz(axis=0) > 0].toarray()
        X = np.asarray(X, dtype=np.float64)
        if varianza:
            X = pca(X, varianza)  # el mismo PCA por abreviatura que en la comparativa
        etiquetas = mejor_ejecucion(X, k, args.metrica, args.init, semillas)
        color_cluster = emparejar(sentidos, etiquetas)

        # Con coseno solo importa la dirección: se dibujan los vectores normalizados, como los ve el K-Means
        P = proyeccion_2d(METRICAS[args.metrica].format_data(X))
        pintar(fila[0], P, indice_sentido, f"{metodo} ({X.shape[1]} dims): sentidos reales")
        pintar(fila[1], P, [color_cluster[e] for e in etiquetas],
               f"{metodo}: nuestro K-Means  (ARI {ari(sentidos, etiquetas):.2f} · NMI {nmi(sentidos, etiquetas):.2f} · pureza {pureza(sentidos, etiquetas):.0%})")

    leyenda = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          color=COLORES[i] if i < len(COLORES) else GRIS, label=s)
               for i, s in enumerate(nombres_sentidos[:len(COLORES)])]
    if k > len(COLORES):
        leyenda.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=GRIS, label="otros sentidos"))
    fig.suptitle(f"{abrev}: {len(grupo)} contextos, {k} sentidos  ({args.metrica}, {args.init})",
                 color=TEXTO, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / len(variantes)))
    fig.legend(handles=leyenda, loc="upper center", ncol=min(3, len(leyenda)), frameon=False,
               fontsize=9, labelcolor=TEXTO_SUAVE, bbox_to_anchor=(0.5, 0.0))
    ruta = f"{salida}/clusters_{re.sub(r'[^A-Za-z0-9_-]', '_', abrev)}.png"
    fig.savefig(ruta, dpi=120, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  {ruta}")


def figura_ari(comparativa, salida):
    """
    Barras horizontales con el ARI medio de cada representación, de mejor a peor. Si se probaron varias métricas e
    inicializaciones, cada representación sale una vez, con su mejor combinación (con todas no se leería nada), y
    aparte se dibujan todas las combinaciones de la representación ganadora.
    """
    if comparativa[["metrica", "init"]].drop_duplicates().shape[0] > 1:
        figura_configuraciones(comparativa, salida)
        comparativa = comparativa.sort_values("ARI", ascending=False).drop_duplicates("metodo")
    comparativa = comparativa.sort_values("ARI")
    # Si todas las filas usan la misma métrica e inicialización, basta con decirlo una vez en el título
    una_sola = comparativa[["metrica", "init"]].drop_duplicates().shape[0] == 1
    nombres = [m if una_sola else f"{m}  ({me}, {i})" for m, me, i in comparativa[["metodo", "metrica", "init"]].values]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(comparativa) + 1.2))
    fig.patch.set_facecolor(FONDO), ax.set_facecolor(FONDO)
    ax.barh(nombres, comparativa["ARI"], color=COLORES[0], height=0.7)
    for y, valor in enumerate(comparativa["ARI"]):
        ax.text(valor + 0.003, y, f"{valor:.3f}", va="center", fontsize=8, color=TEXTO_SUAVE)
    ax.set_xlabel("ARI medio sobre abreviaturas (1 = clusters idénticos a los sentidos, 0 = azar)",
                  color=TEXTO_SUAVE, fontsize=9)
    titulo = "¿Qué representación agrupa mejor con nuestro K-Means?"
    if una_sola:
        titulo += f"  ({comparativa['metrica'].iloc[0]}, {comparativa['init'].iloc[0]})"
    ax.set_title(titulo, color=TEXTO, loc="left")
    ax.tick_params(colors=TEXTO_SUAVE, labelsize=8)
    ax.grid(axis="x", color="#e6e5e0"), ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{salida}/ari_por_metodo.png", dpi=120, facecolor=FONDO)
    plt.close(fig)
    print(f"  {salida}/ari_por_metodo.png")


def figura_configuraciones(comparativa, salida):
    """ARI de cada métrica e inicialización con la representación ganadora: el mejor de 10 semillas y una sola."""
    ganadora = comparativa.sort_values("ARI", ascending=False)["metodo"].iloc[0]
    filas = comparativa[comparativa["metodo"] == ganadora].sort_values("ARI")
    nombres = [f"{me}, {i}" for me, i in filas[["metrica", "init"]].values]
    y = np.arange(len(filas))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(filas) + 1.8))
    fig.patch.set_facecolor(FONDO), ax.set_facecolor(FONDO)
    ax.barh(y + 0.18, filas["ARI"], height=0.36, color=COLORES[0], label="mejor de 10 semillas (menor inercia)")
    ax.barh(y - 0.18, filas["ARI_medio"], height=0.36, color=COLORES[1], label="una sola ejecución (media)")
    for i, (a, m) in enumerate(filas[["ARI", "ARI_medio"]].values):
        ax.text(a + 0.003, i + 0.18, f"{a:.4f}", va="center", fontsize=8, color=TEXTO_SUAVE)
        ax.text(m + 0.003, i - 0.18, f"{m:.3f}", va="center", fontsize=8, color=TEXTO_SUAVE)
    ax.set_yticks(y, nombres)
    ax.set_xlabel("ARI medio sobre abreviaturas", color=TEXTO_SUAVE, fontsize=9)
    ax.set_title(f"Métrica e inicialización de nuestro K-Means con {ganadora}", color=TEXTO, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    ax.tick_params(colors=TEXTO_SUAVE, labelsize=8)
    ax.grid(axis="x", color="#e6e5e0"), ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{salida}/ari_por_configuracion.png", dpi=120, facecolor=FONDO)
    plt.close(fig)
    print(f"  {salida}/ari_por_configuracion.png")


def elegir_abreviaturas(df, ruta_detalle):
    """Mejor, mediana y peor abreviatura del mejor método según la comparativa (o 3 al azar si no existe)."""
    candidatas = df.groupby("abreviatura")["sentido"].nunique().between(2, len(COLORES))
    candidatas = set(candidatas[candidatas].index)
    if not os.path.exists(ruta_detalle):
        return list(np.random.default_rng(0).choice(sorted(candidatas), 3, replace=False))
    detalle = pd.read_csv(ruta_detalle)
    detalle = detalle[detalle["abreviatura"].isin(candidatas)]
    mejor = detalle.groupby(["metodo", "metrica", "init"])["ARI"].mean().idxmax()
    filas = detalle[(detalle[["metodo", "metrica", "init"]] == mejor).all(axis=1)].sort_values("ARI")
    return [filas["abreviatura"].iloc[i] for i in (-1, len(filas) // 2, 0)]


def main():
    parser = argparse.ArgumentParser(description="Gráficos de los clusters de nuestro K-Means")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--abreviaturas", nargs="+", default=None)
    parser.add_argument("--metodos", nargs="+", default=None,
                        help="Representaciones o variantes (p. ej. bert_media+pca80%%); "
                             "por defecto, el ganador, su versión sin PCA, w2v y tfidf")
    parser.add_argument("--metrica", default=None, choices=list(METRICAS), help="Por defecto, la de la ganadora")
    parser.add_argument("--init", default=None, choices=list(INICIALIZACIONES), help="Por defecto, la de la ganadora")
    parser.add_argument("--n-semillas", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    salida = f"{args.representaciones}/graficos"
    os.makedirs(salida, exist_ok=True)

    ruta_resumen = f"{args.representaciones}/comparativa_kmeans_propio.csv"
    if os.path.exists(ruta_resumen):
        figura_ari(pd.read_csv(ruta_resumen), salida)
        # Métrica e inicialización de la ganadora, si no se dan
        ganadora = pd.read_csv(ruta_resumen).iloc[0]
        args.metrica, args.init = args.metrica or ganadora["metrica"], args.init or ganadora["init"]
    args.metrica, args.init = args.metrica or "coseno", args.init or "kmeans++"

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido"])
    abreviaturas = args.abreviaturas or elegir_abreviaturas(
        df, f"{args.representaciones}/comparativa_kmeans_propio_por_abreviatura.csv")
    desconocidas = set(abreviaturas) - set(df["abreviatura"])
    if desconocidas:
        raise SystemExit(f"Abreviaturas que no están en instancias.csv: {', '.join(desconocidas)}")
    variantes = args.metodos or metodos_por_defecto(ruta_resumen)
    if variantes is None:
        raise SystemExit("No hay comparativa_kmeans_propio.csv: indica las representaciones con --metodos")
    # Se carga también bow para quitar las mismas instancias vacías que la comparativa (son vacías en bow/tfidf/lsa)
    origenes = list(dict.fromkeys([separar_variante(v)[0] for v in variantes] + ["bow"]))
    matrices = cargar_matrices(args.representaciones, origenes)
    faltan = {separar_variante(v)[0] for v in variantes} - set(matrices)
    if faltan:
        raise SystemExit(f"No existen las matrices X_ de: {', '.join(faltan)}")
    semillas = [args.semilla + i for i in range(args.n_semillas)]

    # Igual que en la comparativa: fuera las instancias con vector vacío en alguna representación
    vacias = np.zeros(len(df), dtype=bool)
    for M in matrices.values():
        vacias |= (M.getnnz(axis=1) == 0) if sparse.issparse(M) else ~np.any(M, axis=1)
    df = df[~vacias]

    print(f"Abreviaturas: {', '.join(abreviaturas)}; representaciones: {', '.join(variantes)}")
    for abrev in abreviaturas:
        figura_abreviatura(abrev, df[df["abreviatura"] == abrev], matrices, variantes, args, semillas, salida)


if __name__ == "__main__":
    main()
