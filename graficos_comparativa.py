"""
Gráficos para el informe: elección del PCA y comparación de las vectorizaciones.

Parte de los resultados de comparativa_kmeans_propio.py (no vuelve a entrenar el K-Means) y dibuja:
    - pca_varianza_acumulada.png: varianza explicada acumulada según el nº de componentes del PCA, para la
      representación ganadora. El PCA se hace por abreviatura (cada una tiene sus contextos), así que hay una
      curva por abreviatura y se resalta la mediana; a la derecha, cuántas componentes se quedan con el 80 %.
    - pca_ari_por_varianza.png: ARI de cada representación sin PCA y quedándose con el 95, 90 y 80 % de la varianza.
    - vectorizaciones_metricas.png: ARI, NMI y pureza de la mejor variante de cada representación.
    - vectorizaciones_ari_distribucion.png: reparto del ARI por abreviatura (diagrama de cajas).
    - vectorizaciones_ari_por_sentidos.png: ARI medio según el nº de sentidos de la abreviatura.

Uso (después de comparativa_kmeans_propio.py):
    python graficos_comparativa.py
    python graficos_comparativa.py --metodo-pca st --varianza 0.9

Resultados en <representaciones>/graficos/
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse

from comparativa_kmeans_propio import cargar_matrices
from graficos_clusters import COLORES, FONDO, GRIS, TEXTO, TEXTO_SUAVE
import matplotlib.pyplot as plt

# Cada representación se colorea según su familia, para que se vea qué tipo de vectorización funciona mejor
FAMILIAS = {
    "bert_media": "contextual", "bert_abrev": "contextual", "st": "contextual",
    "w2v": "embeddings estáticos", "d2v": "embeddings estáticos",
    "lsa": "temas", "lda": "temas",
    "bow": "dispersa", "tfidf": "dispersa",
}
COLOR_FAMILIA = {"contextual": COLORES[0], "embeddings estáticos": COLORES[2], "temas": COLORES[3],
                 "dispersa": COLORES[1]}
REJILLA = "#e6e5e0"


def separar_variante(variante):
    """'bert_media+pca80%' -> ('bert_media', 80); 'w2v' -> ('w2v', 100)."""
    origen, _, porcentaje = re.match(r"([^+]+)(\+pca(\d+)%)?$", variante).groups()
    return origen, int(porcentaje) if porcentaje else 100


def preparar_ejes(fig, ax, titulo, subtitulo=None):
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.set_title(titulo, color=TEXTO, loc="left", pad=22 if subtitulo else 6)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, color=TEXTO_SUAVE, fontsize=8.5)
    ax.tick_params(colors=TEXTO_SUAVE, labelsize=8.5)
    ax.grid(color=REJILLA)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)
    print(f"  {ruta}")


def varianza_acumulada(X):
    """Fracción de varianza que explican las 1, 2, 3... primeras componentes (mismo cálculo que pca())."""
    S = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    return np.cumsum(S ** 2) / np.sum(S ** 2)


def figura_varianza_pca(directorio, origen, varianza, salida, max_componentes):
    """Curvas de varianza acumulada por abreviatura + histograma de componentes elegidas."""
    df = pd.read_csv(f"{directorio}/instancias.csv", usecols=["abreviatura"])
    # Igual que en la comparativa: fuera las instancias con vector vacío (lo son en bow)
    matrices = cargar_matrices(directorio, [origen, "bow"])
    vacias = np.zeros(len(df), dtype=bool)
    for M in matrices.values():
        vacias |= (M.getnnz(axis=1) == 0) if sparse.issparse(M) else ~np.any(M, axis=1)
    M = matrices[origen]
    df = df[~vacias]

    curvas, elegidas = [], []
    for _, grupo in df.groupby("abreviatura"):
        X = M[grupo.index.to_numpy()]
        X = np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=np.float64)
        acumulada = varianza_acumulada(X)
        elegidas.append(int(np.searchsorted(acumulada, varianza) + 1))
        # Las abreviaturas con pocos contextos llegan antes al 100 %: se rellena para poder hacer la mediana
        curva = np.ones(max_componentes)
        n = min(len(acumulada), max_componentes)
        curva[:n] = acumulada[:n]
        curvas.append(curva)
    curvas, elegidas = np.array(curvas) * 100, np.array(elegidas)
    x = np.arange(1, max_componentes + 1)
    mediana = np.median(curvas, axis=0)

    fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    preparar_ejes(fig, ax, f"Varianza explicada acumulada por el PCA ({origen})",
                  f"Una curva gris por abreviatura ({len(curvas)}); en azul, la mediana")
    muestra = np.random.default_rng(42).choice(len(curvas), min(300, len(curvas)), replace=False)
    for i in muestra:
        ax.plot(x, curvas[i], color=GRIS, alpha=0.15, linewidth=0.7)
    ax.fill_between(x, np.percentile(curvas, 25, axis=0), np.percentile(curvas, 75, axis=0),
                    color=COLORES[0], alpha=0.18, linewidth=0, label="50 % central de las abreviaturas")
    ax.plot(x, mediana, color=COLORES[0], linewidth=2.2, label="mediana")
    for umbral in (80, 90, 95):
        ax.axhline(umbral, color=TEXTO_SUAVE if umbral == round(varianza * 100) else GRIS,
                   linestyle="--", linewidth=1)
        ax.text(max_componentes, umbral - 0.5, f"{umbral} %", ha="right", va="top", fontsize=8, color=TEXTO_SUAVE)
    # Punto en el que la mediana cruza el umbral elegido
    corte = int(np.searchsorted(mediana, varianza * 100) + 1)
    ax.plot([corte], [mediana[corte - 1]], "o", color=COLORES[1], zorder=5)
    ax.annotate(f"{round(varianza * 100)} % de la varianza\n≈ {corte} componentes (mediana)",
                (corte, mediana[corte - 1]), xytext=(corte + 8, mediana[corte - 1] - 22), fontsize=8.5,
                color=TEXTO, arrowprops={"arrowstyle": "-", "color": TEXTO_SUAVE})
    ax.text(42, 30, "Con n contextos hay como mucho n − 1 componentes:\nlas abreviaturas de 2 sentidos "
            "(40 contextos)\nllegan al 100 % con 39", fontsize=8, color=TEXTO_SUAVE)
    ax.set_xlim(1, max_componentes)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Número de componentes del PCA", color=TEXTO_SUAVE, fontsize=9)
    ax.set_ylabel("Varianza explicada acumulada (%)", color=TEXTO_SUAVE, fontsize=9)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.08), fontsize=8, frameon=False, labelcolor=TEXTO_SUAVE)

    preparar_ejes(fig, ax_hist, f"Componentes con el {round(varianza * 100)} % de la varianza",
                  f"media {elegidas.mean():.1f}, mediana {np.median(elegidas):.0f} (de {M.shape[1]} dimensiones)")
    ax_hist.hist(elegidas, bins=np.arange(0.5, np.percentile(elegidas, 99) + 1.5, 2), color=COLORES[0])
    ax_hist.axvline(np.median(elegidas), color=COLORES[1], linewidth=1.5)
    ax_hist.set_xlabel("Componentes que se quedan", color=TEXTO_SUAVE, fontsize=9)
    ax_hist.set_ylabel("Abreviaturas", color=TEXTO_SUAVE, fontsize=9)
    ax_hist.grid(axis="x", visible=False)
    guardar(fig, f"{salida}/pca_varianza_acumulada.png")


def separar_etiquetas(valores, distancia):
    """Altura de la etiqueta de cada línea: su valor final, bajándola si pisa a la de encima."""
    posiciones, ultima = {}, None
    for nombre, valor in valores.sort_values(ascending=False).items():
        posiciones[nombre] = valor if ultima is None else min(valor, ultima - distancia)
        ultima = posiciones[nombre]
    return posiciones


def figura_ari_por_varianza(comparativa, salida):
    """Una línea por representación: ARI sin PCA (100 %) y con el 95, 90 y 80 % de la varianza."""
    # Si se probaron varias distancias e inicializaciones, cada punto es la mejor de su variante
    c = comparativa.sort_values("ARI", ascending=False).drop_duplicates("metodo").copy()
    c[["origen", "porcentaje"]] = [separar_variante(m) for m in c["metodo"]]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    preparar_ejes(fig, ax, "¿Cuánta varianza conservar con el PCA?",
                  "ARI medio de nuestro K-Means según el % de varianza que se queda el PCA (100 % = sin PCA); "
                  "mejor distancia e inicialización")
    orden = c.groupby("origen")["ARI"].max().sort_values(ascending=False).index
    ganador = orden[0]
    posiciones = separar_etiquetas(c[c["porcentaje"] == 80].set_index("origen")["ARI"], 0.017)
    for origen in orden:
        filas = c[c["origen"] == origen].sort_values("porcentaje", ascending=False)
        color = COLOR_FAMILIA[FAMILIAS.get(origen, "dispersa")]
        ancho = 2.6 if origen == ganador else 1.4
        ax.plot(filas["porcentaje"], filas["ARI"], "-o", color=color, linewidth=ancho, markersize=4 + ancho)
        # El eje x va al revés (de sin PCA a 80 %), así que restar es ir a la derecha
        ax.text(79, posiciones[origen], origen, va="center", fontsize=8, color=color)
    ax.set_xticks([100, 95, 90, 80], ["sin PCA", "95 %", "90 %", "80 %"])
    ax.set_xlim(101, 75.5)
    ax.set_xlabel("Varianza que conserva el PCA (por abreviatura)", color=TEXTO_SUAVE, fontsize=9)
    ax.set_ylabel("ARI medio", color=TEXTO_SUAVE, fontsize=9)
    leyenda_familias(ax, orden, loc="center left")
    guardar(fig, f"{salida}/pca_ari_por_varianza.png")


def leyenda_familias(ax, origenes, loc="lower left"):
    presentes = dict.fromkeys(FAMILIAS.get(o, "dispersa") for o in origenes)
    manejadores = [plt.Line2D([], [], color=COLOR_FAMILIA[f], marker="s", linestyle="", markersize=8)
                   for f in presentes]
    ax.legend(manejadores, list(presentes), loc=loc, fontsize=8, frameon=False, labelcolor=TEXTO_SUAVE)


def mejores_variantes(comparativa):
    """La variante con más ARI de cada representación, de mejor a peor."""
    c = comparativa.copy()
    c["origen"] = [separar_variante(m)[0] for m in c["metodo"]]
    return c.sort_values("ARI", ascending=False).drop_duplicates("origen")


def filas_de(mejores, por_abreviatura):
    """Las filas por abreviatura de cada mejor variante, con su misma distancia e inicialización."""
    return por_abreviatura.merge(mejores[["metodo", "metrica", "init"]], on=["metodo", "metrica", "init"])


def figura_metricas(mejores, salida):
    """Barras agrupadas con ARI, NMI y pureza de la mejor variante de cada representación."""
    metricas = [("ARI", COLORES[0]), ("NMI", COLORES[6]), ("Pureza", COLORES[3])]
    x = np.arange(len(mejores))
    ancho = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.8))
    preparar_ejes(fig, ax, "Comparación de las vectorizaciones con nuestro K-Means",
                  "Mejor variante de cada representación (mejor distancia e inicialización; k = nº de sentidos); "
                  "media sobre las abreviaturas")
    for i, (metrica, color) in enumerate(metricas):
        barras = ax.bar(x + (i - 1) * ancho, mejores[metrica], ancho, color=color, label=metrica)
        for barra, valor in zip(barras, mejores[metrica]):
            ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.008, f"{valor:.2f}", ha="center",
                    fontsize=6.5, color=TEXTO_SUAVE)
    ax.set_xticks(x, [m.replace("+", "\n+") for m in mejores["metodo"]], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor de la métrica (1 = perfecto)", color=TEXTO_SUAVE, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=3, fontsize=8.5, frameon=False, labelcolor=TEXTO_SUAVE)
    guardar(fig, f"{salida}/vectorizaciones_metricas.png")


def figura_distribucion(mejores, por_abreviatura, salida):
    """Diagrama de cajas del ARI por abreviatura: además de la media, muestra lo estable que es cada método."""
    por_abreviatura = filas_de(mejores, por_abreviatura)
    datos = [por_abreviatura.loc[por_abreviatura["metodo"] == m, "ARI"].to_numpy() for m in mejores["metodo"]]
    colores = [COLOR_FAMILIA[FAMILIAS.get(o, "dispersa")] for o in mejores["origen"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    preparar_ejes(fig, ax, "Reparto del ARI por abreviatura",
                  f"Mejor variante de cada representación; {len(datos[0])} abreviaturas; el rombo es la media")
    cajas = ax.boxplot(datos, widths=0.55, patch_artist=True, showfliers=False, showmeans=True,
                       medianprops={"color": TEXTO, "linewidth": 1.4},
                       meanprops={"marker": "D", "markerfacecolor": FONDO, "markeredgecolor": TEXTO,
                                  "markersize": 4.5},
                       whiskerprops={"color": TEXTO_SUAVE}, capprops={"color": TEXTO_SUAVE})
    for caja, color in zip(cajas["boxes"], colores):
        caja.set(facecolor=color, alpha=0.75, edgecolor=color)
    ax.set_xticks(np.arange(1, len(datos) + 1), [m.replace("+", "\n+") for m in mejores["metodo"]], fontsize=8)
    ax.axhline(0, color=GRIS, linewidth=1)
    ax.set_ylabel("ARI de la abreviatura", color=TEXTO_SUAVE, fontsize=9)
    ax.grid(axis="x", visible=False)
    leyenda_familias(ax, mejores["origen"], loc="upper right")
    guardar(fig, f"{salida}/vectorizaciones_ari_distribucion.png")


def figura_por_sentidos(mejores, por_abreviatura, salida, max_sentidos=8):
    """ARI medio según el nº de sentidos de la abreviatura (las de más de max_sentidos van juntas)."""
    p = filas_de(mejores, por_abreviatura)
    p["grupo"] = p["n_sentidos"].clip(upper=max_sentidos)
    tabla = p.groupby(["metodo", "grupo"])["ARI"].mean().unstack()
    cuantas = p[p["metodo"] == mejores["metodo"].iloc[0]].groupby("grupo").size()
    fig, ax = plt.subplots(figsize=(9, 5))
    preparar_ejes(fig, ax, "ARI según el número de sentidos de la abreviatura",
                  "Mejor variante de cada representación; entre paréntesis, cuántas abreviaturas hay de cada tipo")
    posiciones = separar_etiquetas(tabla.iloc[:, -1], 0.017)
    for metodo, origen in mejores[["metodo", "origen"]].values:
        color = COLOR_FAMILIA[FAMILIAS.get(origen, "dispersa")]
        ancho = 2.6 if metodo == mejores["metodo"].iloc[0] else 1.3
        ax.plot(tabla.columns, tabla.loc[metodo], "-o", color=color, linewidth=ancho, markersize=3 + ancho)
        ax.text(tabla.columns[-1] + 0.2, posiciones[metodo], metodo, va="center", fontsize=7.5, color=color)
    ax.set_xticks(tabla.columns, [f"{g}{'+' if g == max_sentidos else ''}\n({cuantas.get(g, 0)})"
                                  for g in tabla.columns])
    ax.set_xlim(tabla.columns[0] - 0.3, tabla.columns[-1] + 1.6)
    ax.set_xlabel("Número de sentidos", color=TEXTO_SUAVE, fontsize=9)
    ax.set_ylabel("ARI medio", color=TEXTO_SUAVE, fontsize=9)
    guardar(fig, f"{salida}/vectorizaciones_ari_por_sentidos.png")


def main():
    parser = argparse.ArgumentParser(description="Gráficos del PCA y de la comparación de vectorizaciones")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--salida", default=None, help="Carpeta de los gráficos (por defecto <representaciones>/graficos)")
    parser.add_argument("--metodo-pca", default=None,
                        help="Representación de la curva de varianza; por defecto, la ganadora de la comparativa")
    parser.add_argument("--varianza", type=float, default=None,
                        help="Umbral que se marca en la curva; por defecto, el de la variante ganadora")
    parser.add_argument("--max-componentes", type=int, default=100)
    args = parser.parse_args()
    salida = args.salida or f"{args.representaciones}/graficos"
    os.makedirs(salida, exist_ok=True)

    ruta = f"{args.representaciones}/comparativa_kmeans_propio.csv"
    if not os.path.exists(ruta):
        raise SystemExit("No hay comparativa_kmeans_propio.csv: ejecuta antes comparativa_kmeans_propio.py")
    comparativa = pd.read_csv(ruta)
    por_abreviatura = pd.read_csv(f"{args.representaciones}/comparativa_kmeans_propio_por_abreviatura.csv")
    mejores = mejores_variantes(comparativa)

    figura_ari_por_varianza(comparativa, salida)
    figura_metricas(mejores, salida)
    figura_distribucion(mejores, por_abreviatura, salida)
    figura_por_sentidos(mejores, por_abreviatura, salida)

    origen, porcentaje = separar_variante(comparativa["metodo"].iloc[0])
    origen = args.metodo_pca or origen
    varianza = args.varianza or (porcentaje / 100 if porcentaje < 100 else 0.8)
    figura_varianza_pca(args.representaciones, origen, varianza, salida, args.max_componentes)


if __name__ == "__main__":
    main()
