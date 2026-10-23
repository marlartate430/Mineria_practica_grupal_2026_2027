"""
Visualiza paso a paso lo que hace NUESTRO K-Means (modeloKMeans.py) con los contextos de una abreviatura,
y qué pasa cuando llega una instancia nueva.

No repite el algoritmo: ejecuta el fit() del grupo tal cual, pero con una métrica coseno que apunta los centroides
cada vez que se usa. Así se ven los centroides que elige kmeans_plus_plus (kmeans_init.py) y los de cada vuelta
del bucle de fit(). Antes de agrupar se quita una frase de cada sentido: son las "instancias nuevas", que luego se
asignan con predict() al centroide más cercano.

La representación es la ganadora de la comparativa (bert_media + PCA al 80 %). Se usa la inicialización
kmeans++ aunque en la comparativa completa ganó random por muy poco (ARI 0,6082 frente a 0,6077): con random los
centroides iniciales son solo puntos al azar, y kmeans++ deja ver mejor cómo se eligen. El PCA se ajusta solo con las
frases de entrenamiento y se guarda (media y componentes) para transformar igual las frases nuevas, que es lo
que tendrá que hacer el modelo final. Los dibujos son una proyección a 2D; el K-Means trabaja con todas las
dimensiones del PCA, así que algún punto puede parecer más cerca de otro centroide en el dibujo.

Uso:
    python visualizar_kmeans.py                   # PNS (4 sentidos)
    python visualizar_kmeans.py --abreviatura PBI

Resultados en <representaciones>/graficos/:
    proceso_kmeans_<abrev>.png   resumen en 6 viñetas (para el informe)
    proceso_kmeans_<abrev>.gif   animación paso a paso
    proceso_kmeans_<abrev>.json  datos para la página interactiva (k de 2 a 6)
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from kmeans_init import kmeans_plus_plus
from metrics import CosineMetric
from modeloKMeans import KMeans

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRIS, TEXTO, TEXTO_SUAVE, FONDO = "#b4b3ad", "#0b0b0b", "#52514e", "#fcfcfb"


class MetricaQueApunta(CosineMetric):
    """Distancia coseno del grupo que, además, guarda los centroides con los que se mide cada vez."""

    def __init__(self, n_instancias):
        self.n = n_instancias
        self.apuntes = []

    def get_distance(self, X, centroides):
        # Solo interesan las llamadas con todas las frases: las de kmeans_plus_plus (un centroide nuevo cada vez)
        # y la asignación de cada vuelta de fit() (todos los centroides)
        if len(X) == self.n:
            self.apuntes.append(np.array(centroides, dtype=np.float64, copy=True))
        return CosineMetric.get_distance(X, centroides)


def ajustar_pca(X, varianza):
    """PCA con numpy que devuelve también la media y las componentes, para transformar frases nuevas igual."""
    media = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - media, full_matrices=False)
    n = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), varianza) + 1)
    return media, Vt[:n]


def entrenar_y_apuntar(X, k, semilla):
    """Ejecuta el fit() del grupo y devuelve el modelo, los centroides de kmeans++ y los de cada vuelta."""
    metrica = MetricaQueApunta(len(X))
    km = KMeans(p_cluster_number=k, p_metric=metrica, p_init=kmeans_plus_plus)
    km.fit(X, semilla)
    elegidos = [c[0] for c in metrica.apuntes if len(c) == 1]       # kmeans++ elige un centroide cada vez
    vueltas = [c for c in metrica.apuntes if len(c) == k]           # centroides al empezar cada vuelta
    if len(vueltas) > 1 and np.allclose(vueltas[-1], vueltas[-2]):  # la última es el cálculo de la inercia
        vueltas = vueltas[:-1]
    vueltas.append(km.centroids)
    return km, np.array(elegidos), vueltas


def recortar(texto, abrev, ancho=220):
    """Trozo del contexto centrado en la abreviatura."""
    i = max(texto.find(abrev), 0)
    inicio = max(0, i - ancho // 2)
    return ("…" if inicio else "") + texto[inicio:inicio + ancho] + ("…" if inicio + ancho < len(texto) else "")


def preparar(args):
    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido", "texto"])
    grupo = df[df["abreviatura"] == args.abreviatura]
    if grupo.empty:
        raise SystemExit(f"'{args.abreviatura}' no está en instancias.csv")
    M = np.load(f"{args.representaciones}/X_bert_media.npy", mmap_mode="r")
    X_bert = np.asarray(M[grupo.index.to_numpy()], dtype=np.float64)

    # Una frase de cada sentido se guarda aparte: harán de instancias nuevas
    rng = np.random.default_rng(args.semilla)
    nuevas = np.array([rng.choice(np.flatnonzero(grupo["sentido"].to_numpy() == s))
                       for s in sorted(grupo["sentido"].unique())])
    entreno = np.setdiff1d(np.arange(len(grupo)), nuevas)

    media, componentes = ajustar_pca(X_bert[entreno], args.varianza_pca)
    X = (X_bert[entreno] - media) @ componentes.T
    X_nuevas = (X_bert[nuevas] - media) @ componentes.T

    # Proyección a 2D solo para dibujar, de los vectores normalizados (con coseno solo cuenta la dirección)
    normalizar = CosineMetric.format_data
    media_2d, comp_2d = ajustar_pca(normalizar(X), 1.0)
    comp_2d = comp_2d[:2]
    a_2d = lambda V: (normalizar(np.atleast_2d(V)) - media_2d) @ comp_2d.T

    return {"grupo": grupo, "entreno": entreno, "nuevas": nuevas, "X": X, "X_nuevas": X_nuevas, "a_2d": a_2d,
            "dims": X.shape[1]}


def resultado_k(datos, k, semilla):
    """Todo lo que se dibuja para una k: centroides de kmeans++, vueltas, asignaciones y frases nuevas."""
    X, a_2d = datos["X"], datos["a_2d"]
    km, elegidos, vueltas = entrenar_y_apuntar(X, k, semilla)
    asignaciones = [np.argmin(CosineMetric.get_distance(CosineMetric.format_data(X), C), axis=1) for C in vueltas]
    sentidos = datos["grupo"]["sentido"].to_numpy()
    # Nombre de cada cluster: el sentido mayoritario entre sus frases de entrenamiento
    tabla = pd.crosstab(asignaciones[-1], sentidos[datos["entreno"]])
    nombres = {int(c): fila.idxmax() for c, fila in tabla.iterrows()}
    distancias_nuevas = CosineMetric.get_distance(CosineMetric.format_data(datos["X_nuevas"]), km.centroids)

    # Con coseno, clean_centroids normaliza cada centroide (longitud 1): misma dirección que la media de su grupo,
    # pero al proyectarlo a 2D queda fuera de la nube. Para dibujarlo se usa la media de su grupo sin normalizar,
    # que da exactamente las mismas distancias coseno y sí cae en el centro de sus puntos
    P = a_2d(X)
    vueltas_2d = [a_2d(vueltas[0])]
    for v in range(1, len(vueltas)):
        vueltas_2d.append(np.array([P[asignaciones[v - 1] == c].mean(axis=0) if np.any(asignaciones[v - 1] == c)
                                    else a_2d(vueltas[v][c])[0] for c in range(k)]))
    return {"km": km, "elegidos": elegidos, "vueltas": vueltas, "asignaciones": asignaciones, "nombres": nombres,
            "prediccion": km.predict(datos["X_nuevas"]), "distancias_nuevas": distancias_nuevas,
            "elegidos_2d": a_2d(elegidos), "vueltas_2d": vueltas_2d}


# ---------------------------------------------------------------------------
# Dibujos
# ---------------------------------------------------------------------------

def base(ax, titulo):
    ax.set_facecolor(FONDO)
    ax.set_xticks([]), ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_color("#d6d5d0")
    ax.set_title(titulo, fontsize=10, color=TEXTO, loc="left")


def puntos(ax, P, etiquetas=None):
    colores = GRIS if etiquetas is None else [COLORES[e % len(COLORES)] for e in etiquetas]
    ax.scatter(P[:, 0], P[:, 1], c=colores, s=16, alpha=0.75, linewidths=0, zorder=2)


def centroides(ax, C, contorno=TEXTO):
    for i, (x, y) in enumerate(C):
        ax.scatter(x, y, marker="X", s=190, color=COLORES[i % len(COLORES)], edgecolors=contorno, linewidths=1.3,
                   zorder=5)


def flechas(ax, desde, hasta):
    for (x0, y0), (x1, y1) in zip(desde, hasta):
        if np.hypot(x1 - x0, y1 - y0) > 1e-3:
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                        arrowprops=dict(arrowstyle="->", color=TEXTO_SUAVE, lw=1.3))


def nueva(ax, p, etiqueta_pred, C, texto=None):
    for i, (x, y) in enumerate(C):
        ax.plot([p[0], x], [p[1], y], color=COLORES[i % len(COLORES)], lw=2.2 if i == etiqueta_pred else 0.8,
                ls="-" if i == etiqueta_pred else ":", zorder=3)
    ax.scatter(*p, marker="*", s=420, color=COLORES[etiqueta_pred % len(COLORES)], edgecolors=TEXTO,
               linewidths=1.2, zorder=6)
    if texto:
        ax.text(0.02, 0.02, texto, transform=ax.transAxes, fontsize=8, color=TEXTO_SUAVE, va="bottom", wrap=True)


def limites(axes, P, extra):
    todo = np.vstack([P] + extra)
    margen = (todo.max(axis=0) - todo.min(axis=0)) * 0.08
    for ax in axes:
        ax.set_xlim(todo[:, 0].min() - margen[0], todo[:, 0].max() + margen[0])
        ax.set_ylim(todo[:, 1].min() - margen[1], todo[:, 1].max() + margen[1])


def figura_resumen(datos, r, k, ruta):
    P, P_nuevas = datos["a_2d"](datos["X"]), datos["a_2d"](datos["X_nuevas"])
    V, A = r["vueltas_2d"], r["asignaciones"]
    i_nueva = 0
    fig, ejes = plt.subplots(2, 3, figsize=(15, 9.5))
    fig.patch.set_facecolor(FONDO)
    ejes = ejes.ravel()

    base(ejes[0], "1. Las frases como puntos (aún sin grupos)")
    puntos(ejes[0], P)
    base(ejes[1], f"2. kmeans++ elige {k} centroides iniciales")
    puntos(ejes[1], P)
    centroides(ejes[1], r["elegidos_2d"])
    base(ejes[2], "3. Vuelta 1: cada frase va a su centroide más cercano")
    puntos(ejes[2], P, A[0])
    centroides(ejes[2], V[0])
    base(ejes[3], "4. Vuelta 1: cada centroide se mueve al centro de su grupo")
    puntos(ejes[3], P, A[0])
    flechas(ejes[3], V[0], V[1])
    centroides(ejes[3], V[1])
    base(ejes[4], f"5. Final: tras {len(V) - 1} vueltas los centroides ya no se mueven")
    puntos(ejes[4], P, A[-1])
    centroides(ejes[4], V[-1])
    pred = int(r["prediccion"][i_nueva])
    base(ejes[5], f"6. Llega una frase nueva → centroide más cercano: cluster {pred + 1}")
    puntos(ejes[5], P, A[-1])
    centroides(ejes[5], V[-1])
    nueva(ejes[5], P_nuevas[i_nueva], pred, V[-1],
          "distancias coseno: " + ", ".join(f"c{j + 1} {d:.2f}" for j, d in enumerate(r["distancias_nuevas"][i_nueva])))
    limites(ejes, P, [P_nuevas, r["elegidos_2d"]] + V)

    leyenda = [plt.Line2D([], [], marker="o", ls="", color=COLORES[c % len(COLORES)], markersize=8,
                          label=f"cluster {c + 1} ≈ {r['nombres'].get(c, '—')}") for c in range(k)]
    leyenda.append(plt.Line2D([], [], marker="X", ls="", color=TEXTO_SUAVE, markersize=10, label="centroide"))
    fig.legend(handles=leyenda, loc="lower center", ncol=min(3, len(leyenda)), frameon=False, fontsize=9,
               labelcolor=TEXTO_SUAVE)
    fig.suptitle(f"Nuestro K-Means paso a paso: {datos['abrev']} (bert_media + PCA, {datos['dims']} dims, k = {k}, "
                 f"coseno, kmeans++)", color=TEXTO, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(ruta, dpi=120, facecolor=FONDO)
    plt.close(fig)
    print(f"  {ruta}")


def animacion(datos, r, k, ruta):
    P, P_nuevas = datos["a_2d"](datos["X"]), datos["a_2d"](datos["X_nuevas"])
    V, A = r["vueltas_2d"], r["asignaciones"]
    fotogramas = [("Las frases de la abreviatura como puntos", lambda ax: puntos(ax, P))]
    for i in range(1, k + 1):
        fotogramas.append((f"kmeans++ elige el centroide {i} de {k}" + (" (lejos de los anteriores)" if i > 1 else " (al azar)"),
                           lambda ax, i=i: (puntos(ax, P), centroides(ax, r["elegidos_2d"][:i]))))
    for v in range(len(V) - 1):
        fotogramas.append((f"Vuelta {v + 1}: cada frase va a su centroide más cercano",
                           lambda ax, v=v: (puntos(ax, P, A[v]), centroides(ax, V[v]))))
        fotogramas.append((f"Vuelta {v + 1}: cada centroide se mueve al centro de su grupo",
                           lambda ax, v=v: (puntos(ax, P, A[v]), flechas(ax, V[v], V[v + 1]), centroides(ax, V[v + 1]))))
    fotogramas.append((f"Los centroides ya no se mueven: fin ({len(V) - 1} vueltas)",
                       lambda ax: (puntos(ax, P, A[-1]), centroides(ax, V[-1]))))
    for j in range(len(P_nuevas)):
        pred = int(r["prediccion"][j])
        fotogramas.append((f"Frase nueva {j + 1}: va al centroide más cercano → cluster {pred + 1} "
                           f"(≈ {r['nombres'].get(pred, '—')})",
                           lambda ax, j=j, pred=pred: (puntos(ax, P, A[-1]), centroides(ax, V[-1]),
                                                       nueva(ax, P_nuevas[j], pred, V[-1]))))
    # Las pausas largas en el primer y último fotograma de cada fase se consiguen repitiéndolos
    secuencia = [i for i, _ in enumerate(fotogramas) for _ in range(2)] + [len(fotogramas) - 1] * 3

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor(FONDO)

    def dibujar(n):
        ax.clear()
        titulo, pintar = fotogramas[n]
        base(ax, titulo)
        pintar(ax)
        limites([ax], P, [P_nuevas, r["elegidos_2d"]] + V)

    FuncAnimation(fig, lambda i: dibujar(secuencia[i]), frames=len(secuencia)).save(
        ruta, writer=PillowWriter(fps=1), dpi=90, savefig_kwargs={"facecolor": FONDO})
    plt.close(fig)
    print(f"  {ruta}")


def exportar_json(datos, resultados, ruta):
    """Coordenadas 2D y resultados reales de cada k, para la página interactiva."""
    grupo, a_2d = datos["grupo"], datos["a_2d"]
    textos = grupo["texto"].to_numpy()
    sentidos = grupo["sentido"].to_numpy()
    redondear = lambda A: np.round(np.asarray(A, dtype=float), 4).tolist()
    salida = {
        "abreviatura": datos["abrev"], "dims": datos["dims"], "representacion": "bert_media + PCA 80 %",
        "sentidos": sorted(set(sentidos)),
        "puntos": [{"xy": redondear(p), "sentido": sentidos[i], "texto": recortar(textos[i], datos["abrev"])}
                   for p, i in zip(a_2d(datos["X"]), datos["entreno"])],
        "nuevas": [{"xy": redondear(p), "sentido": sentidos[i], "texto": recortar(textos[i], datos["abrev"])}
                   for p, i in zip(a_2d(datos["X_nuevas"]), datos["nuevas"])],
        "por_k": {},
    }
    for k, r in resultados.items():
        salida["por_k"][k] = {
            "elegidos": redondear(r["elegidos_2d"]),
            "vueltas": [redondear(C) for C in r["vueltas_2d"]],
            "asignaciones": [a.tolist() for a in r["asignaciones"]],
            "nombres": {str(c): n for c, n in r["nombres"].items()},
            "prediccion": r["prediccion"].tolist(),
            "distancias_nuevas": redondear(r["distancias_nuevas"]),
        }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False)
    print(f"  {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Visualiza paso a paso nuestro K-Means en una abreviatura")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--abreviatura", default="PNS")
    parser.add_argument("--k", type=int, default=None, help="Por defecto, el nº de sentidos de la abreviatura")
    parser.add_argument("--varianza-pca", type=float, default=0.8)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    salida = f"{args.representaciones}/graficos"
    os.makedirs(salida, exist_ok=True)

    datos = preparar(args)
    datos["abrev"] = args.abreviatura
    k = args.k or datos["grupo"]["sentido"].nunique()
    resultados = {kk: resultado_k(datos, kk, args.semilla) for kk in sorted({k, *range(2, 7)})}
    nombre = re.sub(r"[^A-Za-z0-9_-]", "_", args.abreviatura)
    r = resultados[k]
    print(f"{args.abreviatura}: {len(datos['X'])} frases de entrenamiento + {len(datos['X_nuevas'])} nuevas, "
          f"{datos['dims']} dims, k = {k}, {len(r['vueltas']) - 1} vueltas")
    figura_resumen(datos, r, k, f"{salida}/proceso_kmeans_{nombre}.png")
    animacion(datos, r, k, f"{salida}/proceso_kmeans_{nombre}.gif")
    exportar_json(datos, resultados, f"{salida}/proceso_kmeans_{nombre}.json")


if __name__ == "__main__":
    main()
