"""
Descripción cuantitativa de los datos (apartado "Análisis de datos" del informe).

Calcula, a partir de lo que deja representacion_vectorial.py en <representaciones>/:
    1. Instancias y atributos: nº de instancias, abreviaturas, sentidos, contextos por sentido y por abreviatura,
       palabras por ventana.
    2. Instancias repetidas (mismo texto) y si alguna repetida tiene sentidos distintos.
    3. Valores perdidos: nulos en instancias.csv, NaN en las matrices y vectores vacíos.
    4. Vocabulario: tokens, tipos, palabras vacías, palabras que salen en un solo contexto y talla del vocabulario
       de BoW y TF-IDF.
    5. Palabras fuera de vocabulario (OOV) en BoW/TF-IDF, BiomedBERT y, si se da --word2vec, word2vec.
    6. Rango de los atributos de cada representación (dimensiones, mínimo, máximo, norma, densidad).

Usa la misma tokenización y las mismas palabras vacías que representacion_vectorial.py.

Uso:
    python analisis_datos.py
    python analisis_datos.py --word2vec ruta/word2vec.kv      # OOV de word2vec (el modelo no se guarda por defecto)
    python analisis_datos.py --sin-bert                       # sin el tokenizador de BiomedBERT (más rápido)

Resultado: lo imprime y lo guarda en <representaciones>/analisis_datos.html (tablas legibles; se abre con el navegador)
y en <representaciones>/analisis_datos.csv (una fila por medida: sección, medida, valor).
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from representacion_vectorial import STOPWORDS, tokenizar

MATRICES = ["bow", "tfidf", "lsa", "lda", "w2v", "d2v", "st", "bert_abrev", "bert_media"]


class Informe:
    """Va imprimiendo las medidas por secciones y las guarda para el CSV."""

    def __init__(self):
        self.filas, self.seccion = [], ""

    def titulo(self, seccion):
        self.seccion = seccion
        print(f"\n{seccion}\n" + "-" * len(seccion))

    def __call__(self, medida, valor):
        if isinstance(valor, float):
            valor = round(valor, 6)
        self.filas.append({"seccion": self.seccion, "medida": medida, "valor": valor})
        print(f"  {medida:<55} {valor}")


def instancias_y_atributos(df, salida):
    salida.titulo("1. Instancias y atributos")
    salida("instancias", len(df))
    salida("atributos de instancias.csv", ", ".join(df.columns))
    salida("abreviaturas", df["abreviatura"].nunique())
    minusculas = df["abreviatura"].str.lower()
    salida("formas de abreviatura en minúsculas", minusculas.nunique())
    salida("sentidos (clases)", df["sentido"].nunique())
    por_sentido = df.groupby(["abreviatura", "sentido"]).size()
    salida("instancias por sentido (mín.)", int(por_sentido.min()))
    salida("instancias por sentido (máx.)", int(por_sentido.max()))
    sentidos = df.groupby("abreviatura")["sentido"].nunique()
    salida("sentidos por abreviatura (media)", float(sentidos.mean()))
    salida("sentidos por abreviatura (mediana)", float(sentidos.median()))
    salida("sentidos por abreviatura (máx.)", int(sentidos.max()))
    por_abrev = df.groupby("abreviatura").size()
    salida("instancias por abreviatura (mín.)", int(por_abrev.min()))
    salida("instancias por abreviatura (máx.)", int(por_abrev.max()))
    salida("abstracts distintos de origen", df["doc"].nunique())
    palabras = df["texto"].str.split().str.len()
    salida("palabras por ventana (media)", float(palabras.mean()))
    salida("palabras por ventana (mediana)", float(palabras.median()))
    salida("palabras por ventana (mín.)", int(palabras.min()))
    salida("palabras por ventana (máx.)", int(palabras.max()))
    salida("ventanas completas (51 palabras, %)", float(100 * (palabras == palabras.max()).mean()))


def repetidas(df, matrices, salida):
    salida.titulo("2. Instancias repetidas")
    dentro = df.duplicated(["abreviatura", "texto"], keep=False)
    salida("contextos repetidos dentro de la misma abreviatura", int(df.duplicated(["abreviatura", "texto"]).sum()))
    salida("contextos repetidos entre abreviaturas", int(df.duplicated("texto").sum()))
    contradictorias = df[dentro].groupby(["abreviatura", "texto"])["sentido"].nunique().gt(1).sum()
    salida("repetidos con sentidos distintos", int(contradictorias))
    salida("apariciones repetidas (mismo abstract y posición)",
           int(df.duplicated(["doc", "posicion", "abreviatura"]).sum()))
    salida("repetidos (% de las instancias)", float(100 * df.duplicated(["abreviatura", "texto"]).mean()))
    if "bert_media" in matrices:
        X = np.ascontiguousarray(matrices["bert_media"])
        filas = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
        _, cuentas = np.unique(filas, return_counts=True)
        salida("vectores repetidos en bert_media (copias de más)", int((cuentas[cuentas > 1] - 1).sum()))


def perdidos(df, matrices, salida):
    salida.titulo("3. Valores perdidos")
    salida("nulos en instancias.csv", int(df.isna().sum().sum()))
    for nombre, X in matrices.items():
        if sparse.issparse(X):
            nan = int(np.isnan(X.data).sum())
            vacias = int((X.getnnz(axis=1) == 0).sum())
        else:
            nan = int(sum(np.isnan(X[i:i + 50_000]).sum() for i in range(0, len(X), 50_000)))
            vacias = int(sum((~np.any(X[i:i + 50_000], axis=1)).sum() for i in range(0, len(X), 50_000)))
        salida(f"{nombre}: NaN / vectores vacíos", f"{nan} / {vacias}")


def vocabulario(df, matrices, salida):
    salida.titulo("4. Vocabulario")
    # Las mismas palabras vacías que representacion_vectorial.py: las del inglés (sin las negaciones) y las
    # propias abreviaturas
    abreviaturas = {a.lower() for a in df["abreviatura"].unique()}
    vacias = set(STOPWORDS) | abreviaturas
    ocurrencias, documentos = Counter(), Counter()
    for texto in df["texto"]:
        tokens = tokenizar(texto)
        ocurrencias.update(tokens)
        documentos.update(set(tokens))
    total = sum(ocurrencias.values())
    salida("tokens en las ventanas", total)
    salida("tipos (palabras distintas)", len(ocurrencias))
    salida("ocurrencias que son palabras vacías o abreviaturas (%)",
           float(100 * sum(n for w, n in ocurrencias.items() if w in vacias) / total))
    utiles = {w: n for w, n in ocurrencias.items() if w not in vacias}
    salida("tipos sin palabras vacías ni abreviaturas", len(utiles))
    unicos = [w for w in utiles if documentos[w] == 1]
    salida("tipos en un único contexto (df = 1)", len(unicos))
    salida("tipos en un único contexto (% de los tipos útiles)", float(100 * len(unicos) / len(utiles)))
    salida("ocurrencias útiles de tipos con df = 1 (%)",
           float(100 * sum(utiles[w] for w in unicos) / sum(utiles.values())))
    for nombre in ("bow", "tfidf"):
        if nombre in matrices:
            salida(f"vocabulario de {nombre}", matrices[nombre].shape[1])
    return ocurrencias, utiles, unicos


def fuera_de_vocabulario(df, utiles, unicos, args, salida):
    salida.titulo("5. Palabras fuera de vocabulario (OOV)")
    salida("BoW/TF-IDF: ocurrencias útiles descartadas por df = 1 (%)",
           float(100 * sum(utiles[w] for w in unicos) / sum(utiles.values())))

    if args.word2vec:
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load(args.word2vec)
        tipos = {w for texto in df["texto"] for w in tokenizar(texto)}
        oov_tipos = sum(w not in kv.key_to_index for w in tipos)
        oov, total, con_oov, todas_oov = 0, 0, 0, 0
        for texto in df["texto"]:
            tokens = tokenizar(texto)
            fuera = sum(w not in kv.key_to_index for w in tokens)
            oov, total = oov + fuera, total + len(tokens)
            con_oov += fuera > 0
            todas_oov += bool(tokens) and fuera == len(tokens)
        salida("word2vec: tipos OOV (%)", float(100 * oov_tipos / len(tipos)))
        salida("word2vec: ocurrencias OOV (%)", float(100 * oov / total))
        salida("word2vec: ventanas con alguna palabra OOV (%)", float(100 * con_oov / len(df)))
        salida("word2vec: ventanas completamente OOV", int(todas_oov))
    else:
        print("  word2vec: el modelo no se guarda en representaciones/; para el OOV usa --word2vec <modelo>")

    if not args.sin_bert:
        from transformers import AutoTokenizer
        tokenizador = AutoTokenizer.from_pretrained(args.modelo_bert)
        unk = tokenizador.unk_token
        subtokens = [tokenizador.tokenize(t) for t in df["texto"]]
        total = sum(map(len, subtokens))
        salida("BiomedBERT: subtokens [UNK] (%)", float(100 * sum(s.count(unk) for s in subtokens) / total))
        longitudes = np.array([len(s) + 2 for s in subtokens])  # + [CLS] y [SEP]
        salida("BiomedBERT: subtokens por ventana (media)", float(longitudes.mean()))
        salida("BiomedBERT: subtokens por ventana (máx.)", int(longitudes.max()))
        salida("BiomedBERT: ventanas truncadas (más de 256)", int((longitudes > 256).sum()))
        partes = np.array([len(tokenizador.tokenize(a)) for a in df["abreviatura"].unique()])
        salida("BiomedBERT: abreviaturas de un solo subtoken (%)", float(100 * (partes == 1).mean()))
        salida("BiomedBERT: subtokens por abreviatura (media)", float(partes.mean()))


def rangos(matrices, semilla, salida):
    salida.titulo("6. Rango de los atributos")
    for nombre, X in matrices.items():
        if sparse.issparse(X):
            salida(f"{nombre}: dimensiones", X.shape[1])
            salida(f"{nombre}: rango", f"[{numero(X.data.min())}; {numero(X.data.max())}] (sin los ceros)")
            salida(f"{nombre}: valores no nulos por fila (media)", float(X.getnnz(axis=1).mean()))
            salida(f"{nombre}: densidad (%)", float(100 * X.nnz / (X.shape[0] * X.shape[1])))
            continue
        minimo, maximo, normas = np.inf, -np.inf, []
        for i in range(0, len(X), 50_000):
            bloque = np.asarray(X[i:i + 50_000], dtype=np.float64)
            minimo, maximo = min(minimo, bloque.min()), max(maximo, bloque.max())
            normas.append(np.linalg.norm(bloque, axis=1))
        salida(f"{nombre}: dimensiones", X.shape[1])
        salida(f"{nombre}: rango", f"[{numero(minimo)}; {numero(maximo)}]")
        salida(f"{nombre}: norma media", float(np.concatenate(normas).mean()))
        if nombre == "lda":
            salida("suma de cada fila de lda (media)", float(np.asarray(X[:50_000]).sum(axis=1).mean()))
    if "bert_media" in matrices:
        # Coseno medio entre pares de contextos elegidos al azar: mide la anisotropía que motiva el PCA
        X = matrices["bert_media"]
        rng = np.random.default_rng(semilla)
        a, b = rng.integers(0, len(X), 20_000), rng.integers(0, len(X), 20_000)
        A, B = np.asarray(X[np.sort(a)], dtype=np.float64), np.asarray(X[np.sort(b)], dtype=np.float64)
        coseno = np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
        salida("coseno medio entre pares de contextos al azar en bert_media", float(coseno.mean()))


def numero(valor):
    """Formato español: 257.500 · 4,06 · 0,000023."""
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if x == int(x) and abs(x) >= 1:
        return f"{int(x):,}".replace(",", ".")
    decimales = 2 if abs(x) >= 1 else max(2, -int(np.floor(np.log10(abs(x)))) + 1) if x else 0
    return f"{x:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


EXPLICACIONES = {
    "1.": "Cuántos datos hay. Cada instancia es una aparición de una abreviatura con su ventana de ±25 palabras; "
          "el sentido es la clase, que solo se usa para evaluar.",
    "2.": "Instancias con el mismo texto. Si dos repetidas tuvieran sentidos distintos, las etiquetas se "
          "contradirían; no pasa nunca.",
    "3.": "Datos que faltan. Un vector vacío es un texto que se queda sin palabras al quitar las palabras vacías "
          "(por ejemplo, el texto es solo «CEA»).",
    "4.": "Palabras de las ventanas. Un token es cada palabra que aparece; un tipo, cada palabra distinta. "
          "El vocabulario de BoW y TF-IDF son las palabras que se quedan como columnas.",
    "5.": "Palabras que un modelo no conoce (out of vocabulary). BoW y TF-IDF crean el vocabulario con estas "
          "mismas ventanas, así que solo descartan las palabras de un único contexto; BiomedBERT trocea las palabras "
          "que no conoce y solo pierde las que acaban como [UNK].",
    "6.": "Cómo son los vectores de cada representación: cuántos números tienen, entre qué valores están y lo "
          "largos que son. En BoW y TF-IDF casi todos los valores son 0.",
}

ESTILO = """<style>
body { font-family: system-ui, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 16px;
       color: #1f1e1d; background: #fbfbf9; line-height: 1.5; }
h1 { font-size: 1.6em; } h2 { font-size: 1.2em; margin-top: 2em; border-bottom: 1px solid #ddd; }
p.nota { color: #5e5d59; margin-top: 0; }
table { border-collapse: collapse; margin: 8px 0 16px; }
th, td { padding: 4px 12px; border-bottom: 1px solid #e6e5e0; text-align: left; }
th { background: #f0efea; } td.valor { text-align: right; font-variant-numeric: tabular-nums; }
</style>"""


def escribir_resumen(filas, ruta):
    """Las mismas medidas que el CSV, en una página HTML con tablas y una explicación por sección."""
    from html import escape
    filas = pd.DataFrame(filas)
    partes_html = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Análisis de datos</title>"
                   f"{ESTILO}</head><body>",
                   "<h1>Análisis de datos: descripción cuantitativa</h1>",
                   "<p class='nota'>Generado por <code>analisis_datos.py</code> a partir de "
                   "<code>instancias.csv</code> y las matrices <code>X_*</code>. Los mismos números están en "
                   "<code>analisis_datos.csv</code>, una fila por medida.</p>"]

    def tabla_html(cabecera, filas_tabla):
        html = "<table><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cabecera) + "</tr>"
        for fila in filas_tabla:
            html += "<tr><td>" + escape(str(fila[0])) + "</td>"
            html += "".join(f"<td class='valor'>{escape(v)}</td>" for v in fila[1:]) + "</tr>"
        return html + "</table>"

    for seccion, grupo in filas.groupby("seccion", sort=False):
        partes_html.append(f"<h2>{escape(seccion)}</h2>")
        explicacion = EXPLICACIONES.get(seccion[:2])
        if explicacion:
            partes_html.append(f"<p class='nota'>{escape(explicacion)}</p>")
        if seccion.startswith(("3.", "6.")):
            # Una fila por representación y una columna por medida
            partes = grupo["medida"].str.split(": ", n=1, expand=True)
            general = grupo[partes[1].isna()]
            tabla = grupo[partes[1].notna()].assign(repr=partes[0], col=partes[1])
            tabla = tabla.pivot(index="repr", columns="col", values="valor").reindex(
                [m for m in MATRICES if m in set(tabla["repr"])])
            orden = ["dimensiones", "rango", "norma media", "valores no nulos por fila (media)", "densidad (%)"]
            tabla = tabla[[c for c in orden if c in tabla.columns] + [c for c in tabla.columns if c not in orden]]
            partes_html.append(tabla_html(["representación"] + list(tabla.columns),
                                          [[n] + ["" if pd.isna(v) else numero(v) for v in fila]
                                           for n, fila in tabla.iterrows()]))
            grupo = general
        if len(grupo):
            partes_html.append(tabla_html(["medida", "valor"],
                                          [[m, numero(v)] for m, v in grupo[["medida", "valor"]].values]))
    partes_html.append("</body></html>")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(partes_html))


def main():
    parser = argparse.ArgumentParser(description="Descripción cuantitativa de los datos de MeDAL")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--word2vec", default=None, help="Modelo word2vec guardado con gensim (para el OOV)")
    parser.add_argument("--modelo-bert", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--sin-bert", action="store_true", help="No calcular el OOV de BiomedBERT")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--solo-resumen", action="store_true",
                        help="Rehacer analisis_datos.html a partir de analisis_datos.csv, sin recalcular nada")
    args = parser.parse_args()

    ruta = f"{args.representaciones}/analisis_datos.csv"
    if args.solo_resumen:
        filas = pd.read_csv(ruta, keep_default_na=False).to_dict("records")
        escribir_resumen(filas, ruta.replace(".csv", ".html"))
        return print(f"Guardado en '{ruta.replace('.csv', '.html')}'")

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", keep_default_na=False, na_values=[""])
    matrices = {}
    for nombre in MATRICES:
        base = f"{args.representaciones}/X_{nombre}"
        if os.path.exists(f"{base}.npz"):
            matrices[nombre] = sparse.load_npz(f"{base}.npz").tocsr()
        elif os.path.exists(f"{base}.npy"):
            matrices[nombre] = np.load(f"{base}.npy", mmap_mode="r")

    salida = Informe()
    instancias_y_atributos(df, salida)
    repetidas(df, matrices, salida)
    perdidos(df, matrices, salida)
    _, utiles, unicos = vocabulario(df, matrices, salida)
    fuera_de_vocabulario(df, utiles, unicos, args, salida)
    rangos(matrices, args.semilla, salida)

    pd.DataFrame(salida.filas).to_csv(ruta, index=False)
    escribir_resumen(salida.filas, ruta.replace(".csv", ".html"))
    print(f"\nGuardado en '{ruta}' y, en tablas legibles, en '{ruta.replace('.csv', '.html')}'")


if __name__ == "__main__":
    main()
