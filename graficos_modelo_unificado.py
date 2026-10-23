"""
Gráficos del modelo unificado (modelo_unificado.py). No vuelve a entrenar: lee el modelo y sus resultados.

Del modelo entero (a partir de modelo_unificado_por_abreviatura.csv):
    - k_elegida_vs_real.png      cuántas abreviaturas tienen cada combinación de k elegida (silueta) y k real
    - ari_por_sentidos.png       % de abreviaturas con la k exacta y ARI de prueba según el nº de sentidos
    - ari_entrena_vs_prueba.png  ARI de entrenamiento frente a prueba de cada abreviatura (¿generaliza?)

De cada abreviatura elegida (clusters_<abreviatura>.png), dos mapas 2D con los mismos puntos:
    - izquierda: coloreados por su sentido REAL
    - derecha:   coloreados por el cluster al que los asigna el modelo (cada cluster con el color del sentido que
                 predomina en él) y con los centroides marcados
Los círculos pequeños son textos de entrenamiento, los rombos con borde, textos de prueba (no vistos al entrenar), y
las aspas, el centro de cada cluster.

Uso (después de modelo_unificado.py):
    python graficos_modelo_unificado.py                              # CEA, PCA y 5-HT
    python graficos_modelo_unificado.py --abreviaturas RA MS

Resultados en modelos_unificado/graficos/
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from comparativa_kmeans_propio import METRICAS
from graficos_clusters import COLORES, FONDO, GRIS, TEXTO, TEXTO_SUAVE, emparejar
from comparativa_kmeans_propio import cargar_matrices
from modelo_unificado import ModeloUnificado, separar_prueba, vectores

BORDE, REJILLA = "#d6d5d0", "#e6e5e0"


def preparar_ejes(ax, titulo):
    ax.set_facecolor(FONDO)
    ax.set_title(titulo, color=TEXTO, loc="left", fontsize=10)
    ax.tick_params(colors=TEXTO_SUAVE, labelsize=8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(BORDE)


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=120, bbox_inches="tight", facecolor=FONDO)
    plt.close(fig)
    print(f"  {ruta}")


def figura_k(r, salida):
    """Mapa de calor: filas = k real, columnas = k elegida por la silueta; la diagonal son los aciertos."""
    filas = [str(k) for k in range(2, 10)] + ["10+"]
    k_real = r["k_real"].clip(upper=10).map(lambda k: "10+" if k >= 10 else str(k))
    tabla = pd.crosstab(k_real, r["k"]).reindex(index=filas, columns=range(2, r["k"].max() + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(FONDO)
    escala = LinearSegmentedColormap.from_list("azul", [FONDO, COLORES[0]])
    imagen = ax.imshow(np.log1p(tabla.values), cmap=escala, aspect="auto")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            n = tabla.values[i, j]
            if n:
                claro = np.log1p(n) > 0.6 * np.log1p(tabla.values.max())
                ax.text(j, i, n, ha="center", va="center", fontsize=7, color=FONDO if claro else TEXTO_SUAVE)
            if filas[i] == str(tabla.columns[j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEXTO, linewidth=1))
    ax.set_xticks(range(tabla.shape[1]), tabla.columns)
    ax.set_yticks(range(tabla.shape[0]), filas)
    ax.set_xlabel("k elegida con la silueta (sin mirar los sentidos)", color=TEXTO_SUAVE, fontsize=9)
    ax.set_ylabel("k real (nº de sentidos)", color=TEXTO_SUAVE, fontsize=9)
    acierto = (r["k"] == r["k_real"]).mean()
    preparar_ejes(ax, f"Número de abreviaturas según su k real y la k elegida  "
                      f"(recuadro = k exacta: {acierto:.0%} de {len(r)} abreviaturas)")
    imagen.set_clim(0, np.log1p(tabla.values.max()))
    guardar(fig, f"{salida}/k_elegida_vs_real.png")


def figura_por_sentidos(r, salida):
    """Dos paneles con las mismas barras: % de k exacta y ARI medio de prueba, por nº de sentidos."""
    grupos = pd.cut(r["k_real"], [1, 2, 3, 4, 5, 9, np.inf], labels=["2", "3", "4", "5", "6–9", "10+"])
    resumen = r.groupby(grupos, observed=True).agg(n=("k", "size"),
                                                   acierto=("k", lambda s: (s == r.loc[s.index, "k_real"]).mean()),
                                                   ari=("ARI_prueba", "mean"))
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.patch.set_facecolor(FONDO)
    for ax, columna, titulo, formato in [(ejes[0], "acierto", "Abreviaturas con la k exacta", "{:.0%}"),
                                         (ejes[1], "ari", "ARI medio en prueba (textos no vistos)", "{:.2f}")]:
        x = np.arange(len(resumen))
        ax.bar(x, resumen[columna], color=COLORES[0], width=0.6)
        for xi, valor in zip(x, resumen[columna]):
            ax.text(xi, valor + 0.01, formato.format(valor), ha="center", va="bottom", fontsize=8, color=TEXTO_SUAVE)
        ax.set_xticks(x, [f"{g}\n({n})" for g, n in zip(resumen.index, resumen["n"])])
        ax.set_xlabel("nº de sentidos de la abreviatura (nº de abreviaturas)", color=TEXTO_SUAVE, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=REJILLA), ax.set_axisbelow(True)
        preparar_ejes(ax, titulo)
    fig.tight_layout()
    guardar(fig, f"{salida}/ari_por_sentidos.png")


def figura_entrena_prueba(r, salida):
    """Un punto por abreviatura; cerca de la diagonal = lo mismo con textos vistos y no vistos."""
    r = r.dropna(subset=["ARI_entrena", "ARI_prueba"])
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    fig.patch.set_facecolor(FONDO)
    ax.plot([-0.3, 1], [-0.3, 1], color=BORDE, linewidth=1, zorder=1)
    ax.scatter(r["ARI_entrena"], r["ARI_prueba"], s=9, color=COLORES[0], alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(r["ARI_entrena"].mean(), r["ARI_prueba"].mean(), s=80, color=COLORES[1], edgecolors=FONDO,
               linewidths=2, zorder=3)
    ax.text(-0.27, 0.97, f"punto grande: media de {r['ARI_entrena'].mean():.2f} en entrenamiento, {r['ARI_prueba'].mean():.2f} en prueba",
            fontsize=8, color=TEXTO, va="top")
    ax.text(-0.27, 0.91, "línea gris: el mismo ARI en los dos", fontsize=8, color=TEXTO_SUAVE, va="top")
    ax.set_xlim(-0.3, 1.02), ax.set_ylim(-0.3, 1.02)
    ax.set_xlabel("ARI con los textos de entrenamiento", color=TEXTO_SUAVE, fontsize=9)
    ax.set_ylabel("ARI con los textos de prueba", color=TEXTO_SUAVE, fontsize=9)
    ax.grid(color=REJILLA), ax.set_axisbelow(True)
    preparar_ejes(ax, f"¿Generaliza? Un punto por abreviatura ({len(r)})")
    guardar(fig, f"{salida}/ari_entrena_vs_prueba.png")


def figura_abreviatura(abrev, modelo, df, M, r, semilla, salida):
    # La misma separación entrenamiento / prueba que modelo_unificado.py (mismo generador y mismo orden)
    rng = np.random.default_rng(semilla)
    for a, g in df[df["abreviatura"].isin(modelo.partes)].groupby("abreviatura"):
        es_prueba = separar_prueba(g, 0.2, rng)
        if a == abrev:
            break
    grupo = df[df["abreviatura"] == abrev]
    entrena, prueba = grupo[~es_prueba], grupo[es_prueba]

    metrica = METRICAS[modelo.metrica]()
    Z = {nombre: metrica.format_data(modelo._transformar(abrev, vectores(M, parte.index.to_numpy())))
         for nombre, parte in [("entrena", entrena), ("prueba", prueba)]}
    etiquetas = {nombre: modelo.asignar(vectores(M, parte.index.to_numpy()),
                                        np.full(len(parte), abrev))
                 for nombre, parte in [("entrena", entrena), ("prueba", prueba)]}
    # Proyección 2D ajustada con los de entrenamiento y aplicada igual a prueba y centroides (solo para dibujar)
    media = Z["entrena"].mean(axis=0)
    _, _, Vt = np.linalg.svd(Z["entrena"] - media, full_matrices=False)
    proyectar = lambda A: (A - media) @ Vt[:2].T
    # Con coseno los centroides están normalizados y al proyectarlos quedan "estirados" hacia fuera de su nube:
    # se dibuja cada uno en el centro de los textos de entrenamiento que tiene asignados
    P_entrena = proyectar(Z["entrena"])
    centroides = {c: P_entrena[etiquetas["entrena"] == c].mean(axis=0) for c in np.unique(etiquetas["entrena"])}

    # Colores: los sentidos más frecuentes primero; cada cluster, con el color del sentido que predomina en él
    nombres = list(grupo["sentido"].value_counts().index)
    color = lambda i: COLORES[i] if i < len(COLORES) else GRIS
    color_cluster = emparejar(entrena["sentido"].to_numpy(), etiquetas["entrena"])
    fila = r.set_index("abreviatura").loc[abrev]

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor(FONDO)
    for ax, por_cluster in zip(ejes, [False, True]):
        for nombre, parte, marcador, tamano in [("entrena", entrena, "o", 14), ("prueba", prueba, "D", 34)]:
            P = proyectar(Z[nombre])
            if por_cluster:
                colores = [color(color_cluster.get(c, len(COLORES))) for c in etiquetas[nombre]]
            else:
                colores = [color(nombres.index(s)) for s in parte["sentido"]]
            ax.scatter(P[:, 0], P[:, 1], c=colores, s=tamano, marker=marcador, alpha=0.8,
                       edgecolors=TEXTO if nombre == "prueba" else "none", linewidths=0.6)
        if por_cluster:
            for c, (x, y) in centroides.items():
                ax.scatter(x, y, s=180, marker="X", color=color(color_cluster.get(c, len(COLORES))),
                           edgecolors=FONDO, linewidths=2, zorder=5)
        ax.set_xticks([]), ax.set_yticks([])
        for borde in ax.spines.values():
            borde.set_color(BORDE)
    ejes[0].set_title(f"Sentidos reales ({len(nombres)})", color=TEXTO, loc="left", fontsize=10)
    ejes[1].set_title(f"Clusters del modelo (k = {int(fila['k'])}, elegida con la silueta)  ·  "
                      f"prueba: ARI {fila['ARI_prueba']:.2f}, pureza {fila['Pureza_prueba']:.0%}",
                      color=TEXTO, loc="left", fontsize=10)

    leyenda = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=color(i), label=s)
               for i, s in enumerate(nombres[:len(COLORES)])]
    if len(nombres) > len(COLORES):
        leyenda.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=GRIS, label="otros sentidos"))
    leyenda += [plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=TEXTO_SUAVE, label="entrenamiento"),
                plt.Line2D([], [], marker="D", linestyle="", markersize=6, color=FONDO, markeredgecolor=TEXTO,
                           label="prueba (no visto)"),
                plt.Line2D([], [], marker="X", linestyle="", markersize=10, color=TEXTO_SUAVE,
                           label="centro del cluster")]
    fig.suptitle(f"{abrev}: {len(entrena)} textos de entrenamiento y {len(prueba)} de prueba",
                 color=TEXTO, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=leyenda, loc="upper center", ncol=min(4, len(leyenda)), frameon=False, fontsize=8,
               labelcolor=TEXTO_SUAVE, bbox_to_anchor=(0.5, 0.02))
    guardar(fig, f"{salida}/clusters_{re.sub(r'[^A-Za-z0-9_-]', '_', abrev)}.png")


def figura_frase(modelo, abrev, entrena, X_entrena, vector, texto, ruta, n_vecinos=5):
    """
    Dónde cae una frase nueva: mapa 2D de los textos de entrenamiento de la abreviatura (color = cluster, con el
    color del sentido que predomina en él), la frase como una estrella unida a sus textos más cercanos, y la
    distancia de la frase a cada centroide.
    """
    metrica = METRICAS[modelo.metrica]()
    Z = metrica.format_data(modelo._transformar(abrev, X_entrena))
    z = metrica.format_data(modelo._transformar(abrev, vector))
    etiquetas = modelo.asignar(X_entrena, np.full(len(X_entrena), abrev))
    distancias = metrica.get_distance(z, modelo.partes[abrev]["centroides"])[0]
    cercania = metrica.get_distance(z, Z)[0]
    elegido = int(np.argmin(distancias))

    # Proyección 2D ajustada con los de entrenamiento (solo para dibujar; el K-Means usa todas las dimensiones)
    media = Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Z - media, full_matrices=False)
    proyectar = lambda A: (A - media) @ Vt[:2].T
    P, p = proyectar(Z), proyectar(z)[0]

    sentidos = entrena["sentido"].to_numpy()
    nombres = list(pd.Series(sentidos).value_counts().index)
    color_cluster = emparejar(sentidos, etiquetas)
    color = lambda c: COLORES[color_cluster[c]] if color_cluster.get(c, len(COLORES)) < len(COLORES) else GRIS
    predomina = {c: nombres[i] for c, i in color_cluster.items()}

    fig, (ax, ax_barras) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.scatter(P[:, 0], P[:, 1], c=[color(c) for c in etiquetas], s=16, alpha=0.75, linewidths=0, zorder=2)
    for c in np.unique(etiquetas):
        cx, cy = P[etiquetas == c].mean(axis=0)
        ax.scatter(cx, cy, s=170, marker="X", color=color(c), edgecolors=FONDO, linewidths=2, zorder=4)
    for i in np.argsort(cercania)[:n_vecinos]:
        ax.plot([p[0], P[i, 0]], [p[1], P[i, 1]], color=TEXTO_SUAVE, linewidth=0.8, linestyle="--", zorder=3)
    ax.scatter(*p, s=420, marker="*", color=color(elegido), edgecolors=TEXTO, linewidths=1.2, zorder=5)
    ax.annotate("tu frase", p, xytext=(10, 10), textcoords="offset points", fontsize=9, color=TEXTO)
    ax.set_xticks([]), ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_color(BORDE)
    ax.set_title(f"Textos de entrenamiento de {abrev} por cluster  (✖ centro del cluster, -- sus {n_vecinos} más "
                 f"cercanos)", color=TEXTO, loc="left", fontsize=10)

    orden = np.argsort(distancias)[::-1]
    y = np.arange(len(orden))
    ax_barras.barh(y, distancias[orden], color=[color(c) for c in orden], height=0.6)
    for yi, c in zip(y, orden):
        ax_barras.text(distancias[c] + 0.02, yi, f"{distancias[c]:.2f}" + ("  ← gana" if c == elegido else ""),
                       va="center", fontsize=8, color=TEXTO if c == elegido else TEXTO_SUAVE,
                       fontweight="bold" if c == elegido else "normal")
    ax_barras.set_yticks(y, [f"cluster {c}: {predomina.get(c, '?')}" for c in orden], fontsize=8)
    ax_barras.set_xlim(0, distancias.max() * 1.35)
    ax_barras.set_xlabel(f"distancia {modelo.metrica} de tu frase al centroide (menos = más parecido)",
                         color=TEXTO_SUAVE, fontsize=9)
    ax_barras.grid(axis="x", color=REJILLA), ax_barras.set_axisbelow(True)
    preparar_ejes(ax_barras, "Distancia a cada cluster")

    corta = texto if len(texto) <= 110 else texto[:107] + "..."
    fig.suptitle(f"\"{corta}\"\n→ asignada al cluster {elegido} (predomina '{predomina.get(elegido, '?')}')",
                 color=TEXTO, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    guardar(fig, ruta)


def main():
    parser = argparse.ArgumentParser(description="Gráficos del modelo unificado")
    parser.add_argument("--modelo", default="modelos_unificado")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--abreviaturas", nargs="+", default=["CEA", "PCA", "5-HT"])
    parser.add_argument("--semilla", type=int, default=42, help="La misma que en modelo_unificado.py")
    args = parser.parse_args()

    ruta_modelo = f"{args.modelo}/modelo_unificado.npz"
    if not os.path.exists(ruta_modelo):
        raise SystemExit(f"No existe '{ruta_modelo}': entrena antes el modelo con 'python modelo_unificado.py'")
    salida = f"{args.modelo}/graficos"
    os.makedirs(salida, exist_ok=True)

    r = pd.read_csv(f"{args.modelo}/modelo_unificado_por_abreviatura.csv")
    print("Gráficos del modelo entero:")
    figura_k(r, salida)
    figura_por_sentidos(r, salida)
    figura_entrena_prueba(r, salida)

    modelo = ModeloUnificado.cargar(ruta_modelo)
    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido"])
    M = cargar_matrices(args.representaciones, [modelo.metodo])[modelo.metodo]  # la misma con la que se entrenó
    print("Gráficos por abreviatura:")
    for abrev in args.abreviaturas:
        if abrev not in modelo.partes:
            print(f"  '{abrev}' no está en el modelo, se salta")
            continue
        figura_abreviatura(abrev, modelo, df, M, r, args.semilla, salida)


if __name__ == "__main__":
    main()
