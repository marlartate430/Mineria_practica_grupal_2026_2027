"""
Asigna textos nuevos a los clusters del modelo unificado (modelo_unificado.py).

Dos formas:
    1. Con una frase escrita a mano (en inglés, como MeDAL): se vectoriza con el mismo BiomedBERT que
       representacion_vectorial.py (bert_media, ventana de ±25 palabras) y se asigna al cluster más cercano de su
       abreviatura.
           python probar_modelo_unificado.py --abreviatura AD --texto "patients with AD showed memory loss and dementia"
    2. Con contextos de MeDAL que el modelo NO vio al entrenar (el 20 % de prueba de modelo_unificado.py): no hace
       falta BERT, se usan los vectores ya calculados.
           python probar_modelo_unificado.py --abreviatura AD --ejemplos 10

Es clustering: el modelo no predice sentidos, asigna cada texto a un cluster. Para poder interpretar los clusters,
se describe cada uno por el sentido que predomina entre los textos de ENTRENAMIENTO asignados a él, y se mira si
el sentido real del texto coincide con ese sentido predominante (los sentidos solo se usan para esto).
"""

import argparse
import re
import string

import numpy as np
import pandas as pd

from modelo_unificado import ModeloUnificado, separar_prueba


def nombres_clusters(modelo, abrev, grupo, X):
    """Cluster -> (sentido predominante entre sus textos de entrenamiento, cuántos de cuántos)."""
    etiquetas = modelo.asignar(X, np.full(len(X), abrev))
    nombres = {}
    for c in range(len(modelo.partes[abrev]["centroides"])):
        sentidos = grupo["sentido"].to_numpy()[etiquetas == c]
        if len(sentidos):
            valores, cuentas = np.unique(sentidos, return_counts=True)
            nombres[c] = (valores[cuentas.argmax()], cuentas.max(), len(sentidos))
        else:
            nombres[c] = ("(vacío)", 0, 0)
    return nombres


def preparar_texto(texto, abrev, w=25):
    """Como las ventanas de MeDAL: sin puntuación, en minúsculas salvo la abreviatura, ±w palabras alrededor."""
    palabras = [p.strip(string.punctuation) for p in texto.split()]
    palabras = [p if p == abrev else p.lower() for p in palabras if p]
    if abrev not in palabras:
        raise SystemExit(f"La abreviatura '{abrev}' no aparece en el texto (tiene que ir tal cual, con mayúsculas)")
    i = palabras.index(abrev)
    return " ".join(palabras[max(0, i - w):i + 1 + w])


def main():
    parser = argparse.ArgumentParser(description="Asignar textos nuevos a los clusters del modelo unificado")
    parser.add_argument("--abreviatura", required=True)
    parser.add_argument("--texto", default=None, help="Frase en inglés que contenga la abreviatura")
    parser.add_argument("--ejemplos", type=int, default=10, help="Contextos de prueba de MeDAL a mostrar")
    parser.add_argument("--modelo", default="modelos_unificado/modelo_unificado.npz")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--modelo-bert", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--semilla", type=int, default=42, help="La misma que en modelo_unificado.py")
    args = parser.parse_args()

    modelo = ModeloUnificado.cargar(args.modelo)
    abrev = args.abreviatura
    if abrev not in modelo.partes:
        parecidas = [a for a in modelo.partes if a.lower() == abrev.lower()]
        raise SystemExit(f"'{abrev}' no está en el modelo" + (f" (¿quizá {', '.join(parecidas)}?)" if parecidas else ""))

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido", "texto"])
    grupo = df[df["abreviatura"] == abrev]
    M = np.load(f"{args.representaciones}/X_bert_media.npy", mmap_mode="r")
    # Misma separación entrenamiento / prueba que modelo_unificado.py (mismo generador y mismo orden)
    rng = np.random.default_rng(args.semilla)
    for a, g in df[df["abreviatura"].isin(modelo.partes)].groupby("abreviatura"):
        es_prueba = separar_prueba(g, 0.2, rng)
        if a == abrev:
            break
    entrena, prueba = grupo[~es_prueba], grupo[es_prueba]
    nombres = nombres_clusters(modelo, abrev, entrena, np.asarray(M[entrena.index.to_numpy()], dtype=np.float64))

    print(f"\n{abrev}: {len(nombres)} clusters ({grupo['sentido'].nunique()} sentidos reales)")
    for c, (sentido, n, total) in nombres.items():
        print(f"  cluster {c}: predomina '{sentido}' ({n} de sus {total} textos de entrenamiento)")

    if args.texto:
        from representacion_vectorial import BertContextual
        ventana = preparar_texto(args.texto, abrev)
        vector = BertContextual(args.modelo_bert).transform([ventana])
        c = int(modelo.asignar(vector.astype(np.float64), [abrev])[0])
        print(f"\nTexto: {ventana}")
        print(f"-> asignado al cluster {c} (predomina '{nombres[c][0]}')")
    else:
        muestra = prueba.sample(min(args.ejemplos, len(prueba)), random_state=args.semilla)
        etiquetas = modelo.asignar(np.asarray(M[muestra.index.to_numpy()], dtype=np.float64),
                                   muestra["abreviatura"].to_numpy())
        coinciden = 0
        print(f"\n{len(muestra)} contextos de prueba (no vistos al entrenar):")
        for (_, fila), c in zip(muestra.iterrows(), etiquetas):
            coincide = nombres[c][0] == fila["sentido"]
            coinciden += coincide
            texto = re.sub(r"\s+", " ", fila["texto"])
            print(f"\n  {'✔' if coincide else '✘'} asignado al cluster {c} (predomina '{nombres[c][0]}')  |  "
                  f"sentido real: {fila['sentido']}")
            print(f"    ...{texto[:160]}...")
        print(f"\nSentido real = sentido predominante de su cluster en {coinciden}/{len(muestra)} textos")


if __name__ == "__main__":
    main()
