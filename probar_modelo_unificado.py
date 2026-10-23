"""
Asigna textos nuevos a los clusters del modelo unificado (modelo_unificado.py).

Antes hay que entrenar el modelo con 'python modelo_unificado.py' (crea modelos_unificado/modelo_unificado.npz).

Dos formas:
    1. Con una frase escrita a mano (en inglés, como MeDAL): se vectoriza igual que en representacion_vectorial.py,
       con la misma representación con la que se entrenó el modelo (BiomedBERT para bert_media / bert_abrev,
       sentence-transformers para st; ventana de ±25 palabras) y se asigna al cluster más cercano de su abreviatura.
           python probar_modelo_unificado.py --abreviatura AD --texto "patients with AD showed memory loss and dementia"
       Con --paso-a-paso enseña cada paso por dentro: la ventana, los tokens, el vector, el PCA de la abreviatura,
       la distancia a cada centroide y los textos de entrenamiento más cercanos a la frase; además guarda una imagen
       con el mapa de la abreviatura y la frase en él (modelos_unificado/graficos/frase_<abreviatura>_<n>.png).
    2. Con contextos de MeDAL que el modelo NO vio al entrenar (el 20 % de prueba de modelo_unificado.py): no hace
       falta BERT, se usan los vectores ya calculados.
           python probar_modelo_unificado.py --abreviatura AD --ejemplos 10

Es clustering: el modelo no predice sentidos, asigna cada texto a un cluster. Para poder interpretar los clusters,
se describe cada uno por el sentido que predomina entre los textos de ENTRENAMIENTO asignados a él, y se mira si
el sentido real del texto coincide con ese sentido predominante (los sentidos solo se usan para esto).
"""

import argparse
import os
import re
import string

import numpy as np
import pandas as pd

from comparativa_kmeans_propio import METRICAS, cargar_matrices
from modelo_unificado import ModeloUnificado, separar_prueba, vectores


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


def vectorizar(ventana, abrev, metodo, args):
    """
    El vector de una ventana nueva con la misma representación que el modelo (solo las contextuales) y los tokens
    en los que la parte el modelo.
    """
    if metodo in ("bert_media", "bert_abrev"):
        from representacion_vectorial import BertContextual
        bert = BertContextual(args.modelo_bert)
        inicio = ventana.split().index(abrev)
        inicio = len(" ".join(ventana.split()[:inicio])) + (1 if inicio else 0)
        media, vector_abrev = bert._pasada([ventana], [(inicio, inicio + len(abrev))])
        return (media if metodo == "bert_media" else vector_abrev), bert.tokenizer.tokenize(ventana)
    if metodo == "st":
        from representacion_vectorial import SentenceTransformerRep
        st = SentenceTransformerRep(args.modelo_st)
        return st.transform([ventana]), st.modelo.tokenizer.tokenize(ventana)
    raise SystemExit(f"--texto no está disponible para '{metodo}' (necesitaría el vocabulario o el modelo con el que "
                     "se calculó la matriz); usa --ejemplos")


def paso_a_paso(modelo, abrev, ventana, tokens, vector, entrena, X_entrena, nombres, n_vecinos=5):
    """Enseña lo que pasa por dentro al asignar una frase nueva: del texto al cluster."""
    metrica = METRICAS[modelo.metrica]()
    parte = modelo.partes[abrev]
    print("\n" + "=" * 70 + "\nPASO A PASO\n" + "=" * 70)
    print(f"\n1. Ventana de ±25 palabras alrededor de '{abrev}' (sin puntuación, en minúsculas salvo la abreviatura):")
    print(f"   {ventana}")
    print(f"\n2. {modelo.metodo} la parte en {len(tokens)} tokens ('##' = trozo pegado al anterior):")
    print(f"   {' | '.join(tokens)}")
    print(f"\n3. Vector de la frase: {vector.shape[1]} números. Los 5 primeros: {np.round(vector[0, :5], 3)}")
    z = modelo._transformar(abrev, vector)
    if parte["media"] is not None:
        print(f"\n4. PCA propio de {abrev}: se resta su media y se proyecta en sus {len(parte['componentes'])} "
              f"componentes -> {z.shape[1]} números. Los 5 primeros: {np.round(z[0, :5], 3)}")
    zf = metrica.format_data(z)
    distancias = metrica.get_distance(zf, parte["centroides"])[0]
    print(f"\n5. Distancia ({modelo.metrica}) a cada uno de los {len(distancias)} centroides de {abrev}:")
    for i, c in enumerate(np.argsort(distancias)):
        print(f"   {'->' if i == 0 else '  '} cluster {c}: {distancias[c]:.3f}   (predomina '{nombres[c][0]}')")
    print(f"   Gana el más cercano: cluster {np.argmin(distancias)}")
    # Instancias de entrenamiento más cercanas a la frase, en el mismo espacio en el que agrupa el K-Means
    Z_entrena = metrica.format_data(modelo._transformar(abrev, X_entrena))
    cercania = metrica.get_distance(zf, Z_entrena)[0]
    etiquetas = modelo.asignar(X_entrena, np.full(len(X_entrena), abrev))
    print(f"\n6. Los {n_vecinos} textos de entrenamiento más parecidos a la frase:")
    for i in np.argsort(cercania)[:n_vecinos]:
        fila = entrena.iloc[i]
        print(f"   distancia {cercania[i]:.3f} · cluster {etiquetas[i]} · sentido real '{fila['sentido']}'")
        print(f"      ...{re.sub(r'\s+', ' ', fila['texto'])[:140]}...")


def main():
    parser = argparse.ArgumentParser(description="Asignar textos nuevos a los clusters del modelo unificado")
    parser.add_argument("--abreviatura", required=True)
    parser.add_argument("--texto", default=None, help="Frase en inglés que contenga la abreviatura")
    parser.add_argument("--ejemplos", type=int, default=10, help="Contextos de prueba de MeDAL a mostrar")
    parser.add_argument("--paso-a-paso", action="store_true", help="Con --texto: enseñar cada paso por dentro")
    parser.add_argument("--modelo", default="modelos_unificado/modelo_unificado.npz")
    parser.add_argument("--representaciones", default="representaciones")
    parser.add_argument("--modelo-bert", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--modelo-st", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semilla", type=int, default=42, help="La misma que en modelo_unificado.py")
    args = parser.parse_args()

    if not os.path.exists(args.modelo):
        raise SystemExit(f"No existe '{args.modelo}': entrena antes el modelo con 'python modelo_unificado.py'")
    modelo = ModeloUnificado.cargar(args.modelo)
    abrev = args.abreviatura
    if abrev not in modelo.partes:
        parecidas = [a for a in modelo.partes if a.lower() == abrev.lower()]
        raise SystemExit(f"'{abrev}' no está en el modelo" + (f" (¿quizá {', '.join(parecidas)}?)" if parecidas else ""))

    df = pd.read_csv(f"{args.representaciones}/instancias.csv", usecols=["abreviatura", "sentido", "texto"])
    grupo = df[df["abreviatura"] == abrev]
    # La misma representación con la que se entrenó el modelo
    M = cargar_matrices(args.representaciones, [modelo.metodo])[modelo.metodo]
    # Misma separación entrenamiento / prueba que modelo_unificado.py (mismo generador y mismo orden)
    rng = np.random.default_rng(args.semilla)
    for a, g in df[df["abreviatura"].isin(modelo.partes)].groupby("abreviatura"):
        es_prueba = separar_prueba(g, 0.2, rng)
        if a == abrev:
            break
    entrena, prueba = grupo[~es_prueba], grupo[es_prueba]
    X_entrena = vectores(M, entrena.index.to_numpy())
    nombres = nombres_clusters(modelo, abrev, entrena, X_entrena)

    print(f"\nModelo entrenado con {modelo.metodo}" + (f" + PCA {modelo.varianza_pca:.0%}" if modelo.varianza_pca else "")
          + f" ({modelo.metrica}, {modelo.init})")
    print(f"\n{abrev}: {len(nombres)} clusters ({grupo['sentido'].nunique()} sentidos reales)")
    for c, (sentido, n, total) in nombres.items():
        print(f"  cluster {c}: predomina '{sentido}' ({n} de sus {total} textos de entrenamiento)")

    if args.texto:
        ventana = preparar_texto(args.texto, abrev)
        vector, tokens = vectorizar(ventana, abrev, modelo.metodo, args)
        vector = vector.astype(np.float64)
        if args.paso_a_paso:
            paso_a_paso(modelo, abrev, ventana, tokens, vector, entrena, X_entrena, nombres)
            from graficos_modelo_unificado import figura_frase
            carpeta = f"{os.path.dirname(args.modelo) or '.'}/graficos"
            os.makedirs(carpeta, exist_ok=True)
            n = 1
            while os.path.exists(f"{carpeta}/frase_{re.sub(r'[^A-Za-z0-9_-]', '_', abrev)}_{n}.png"):
                n += 1
            print("\n7. Imagen:")
            figura_frase(modelo, abrev, entrena, X_entrena, vector, ventana,
                         f"{carpeta}/frase_{re.sub(r'[^A-Za-z0-9_-]', '_', abrev)}_{n}.png")
        c = int(modelo.asignar(vector, [abrev])[0])
        print(f"\nTexto: {ventana}")
        print(f"-> asignado al cluster {c} (predomina '{nombres[c][0]}')")
    else:
        muestra = prueba.sample(min(args.ejemplos, len(prueba)), random_state=args.semilla)
        etiquetas = modelo.asignar(vectores(M, muestra.index.to_numpy()),
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
