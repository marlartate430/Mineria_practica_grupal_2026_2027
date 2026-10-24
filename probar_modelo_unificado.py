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
       con el mapa de la abreviatura y la frase en él (modelos_unificado/demo/demo_<abreviatura>_<n>.png).
       Se pueden probar varias frases de una vez (BERT se carga una sola vez): --texto "frase 1" "frase 2" ... o
       --archivo frases.txt (una por línea; las que empiezan por "#" se saltan). Todas se van añadiendo a
       modelos_unificado/demo/todas_las_pruebas.txt.
    2. Con contextos de MeDAL que el modelo NO vio al entrenar (el 20 % de prueba de modelo_unificado.py): no hace
       falta BERT, se usan los vectores ya calculados.
           python probar_modelo_unificado.py --abreviatura AD --ejemplos 10

Es clustering: el modelo no predice sentidos, asigna cada texto a un cluster. Para poder interpretar los clusters,
se describe cada uno por el sentido que predomina entre los textos de ENTRENAMIENTO asignados a él, y se mira si
el sentido real del texto coincide con ese sentido predominante (los sentidos solo se usan para esto).
"""

import argparse
import glob
import os
import re
import string
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from comparativa_kmeans_propio import METRICAS, cargar_matrices
from modelo_unificado import ModeloUnificado, separar_prueba, vectores


class Duplicar:
    """Escribe a la vez en pantalla y en un archivo (para guardar lo que sale en cada prueba)."""
    def __init__(self, *salidas):
        self.salidas = salidas

    def write(self, texto):
        for salida in self.salidas:
            salida.write(texto)

    def flush(self):
        for salida in self.salidas:
            salida.flush()

    def isatty(self):  # transformers lo pregunta al cargar BERT
        return False


def siguiente_numero(abrev, carpeta):
    """Primer n libre para demo_<abrev>_<n> (así el .txt y la imagen de una misma prueba llevan el mismo n)."""
    nombre, n = re.sub(r"[^A-Za-z0-9_-]", "_", abrev), 1
    while glob.glob(f"{carpeta}/demo_{nombre}_{n}.*"):
        n += 1
    return f"demo_{nombre}_{n}"


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


_cargados = {}  # BERT / Sentence-Transformers se cargan una sola vez aunque se prueben varias frases


def vectorizar(ventana, abrev, metodo, args):
    """
    El vector de una ventana nueva con la misma representación que el modelo (solo las contextuales), los tokens
    en los que la parte el modelo y el tokenizador.
    """
    if metodo in ("bert_media", "bert_abrev"):
        from representacion_vectorial import BertContextual
        if "bert" not in _cargados:
            _cargados["bert"] = BertContextual(args.modelo_bert)
        bert = _cargados["bert"]
        inicio = ventana.split().index(abrev)
        inicio = len(" ".join(ventana.split()[:inicio])) + (1 if inicio else 0)
        media, vector_abrev = bert._pasada([ventana], [(inicio, inicio + len(abrev))])
        return (media if metodo == "bert_media" else vector_abrev), bert.tokenizer.tokenize(ventana), bert.tokenizer
    if metodo == "st":
        from representacion_vectorial import SentenceTransformerRep
        if "st" not in _cargados:
            _cargados["st"] = SentenceTransformerRep(args.modelo_st)
        st = _cargados["st"]
        return st.transform([ventana]), st.modelo.tokenizer.tokenize(ventana), st.modelo.tokenizer
    raise SystemExit(f"--texto no está disponible para '{metodo}' (necesitaría el vocabulario o el modelo con el que "
                     "se calculó la matriz); usa --ejemplos")


def vocabulario(ventana, tokenizer):
    """
    Cómo ve el tokenizador cada palabra. BERT no tiene palabras fuera de vocabulario (OOV) como word2vec: lo que no
    conoce lo parte en trozos ('##'), y solo si ni así puede sale [UNK]. Muchas palabras partidas = texto que el
    modelo ha visto poco (otro idioma, erratas...).
    """
    enteras, partidas, desconocidas = [], [], []
    for palabra in ventana.split():
        trozos = tokenizer.tokenize(palabra)
        if tokenizer.unk_token in trozos:
            desconocidas.append(palabra)
        elif len(trozos) == 1:
            enteras.append(palabra)
        else:
            partidas.append(f"{palabra} -> {' | '.join(trozos)}")
    return enteras, partidas, desconocidas


def paso_a_paso(modelo, abrev, ventana, tokens, tokenizer, vector, entrena, X_entrena, nombres, n_vecinos=5):
    """Enseña lo que pasa por dentro al asignar una frase nueva: del texto al cluster."""
    metrica = METRICAS[modelo.metrica]()
    parte = modelo.partes[abrev]
    print("\n" + "=" * 70 + "\nPASO A PASO\n" + "=" * 70)
    print(f"\n1. Ventana de ±25 palabras alrededor de '{abrev}' (sin puntuación, en minúsculas salvo la abreviatura):")
    print(f"   {ventana}")
    print(f"\n2. {modelo.metodo} la parte en {len(tokens)} tokens ('##' = trozo pegado al anterior):")
    print(f"   {' | '.join(tokens)}")
    enteras, partidas, desconocidas = vocabulario(ventana, tokenizer)
    n = len(ventana.split())
    print(f"   De sus {n} palabras: {len(enteras)} están enteras en el vocabulario ({len(enteras) / n:.0%}), "
          f"{len(partidas)} se parten en trozos ({len(partidas) / n:.0%}) y {len(desconocidas)} son [UNK]")
    for p in partidas:
        print(f"      {p}")
    if desconocidas:
        print(f"      [UNK]: {', '.join(desconocidas)}")
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
    ganador = np.argmin(distancias)
    print(f"\n6. Los {n_vecinos} textos de entrenamiento del cluster {ganador} más parecidos a la frase:")
    for i in [i for i in np.argsort(cercania) if etiquetas[i] == ganador][:n_vecinos]:
        fila = entrena.iloc[i]
        print(f"   distancia {cercania[i]:.3f} · cluster {etiquetas[i]} · sentido real '{fila['sentido']}'")
        print(f"      ...{re.sub(r'\s+', ' ', fila['texto'])[:140]}...")


def resumen_cluster(modelo, abrev, c, vector, entrena, X_entrena, nombres, ruta_por_abreviatura, n_vecinos=5):
    """Métricas del cluster al que se asigna la frase y del modelo de su abreviatura."""
    metrica = METRICAS[modelo.metrica]()
    centroides = modelo.partes[abrev]["centroides"]
    z = metrica.format_data(modelo._transformar(abrev, vector))
    Z = metrica.format_data(modelo._transformar(abrev, X_entrena))
    etiquetas = modelo.asignar(X_entrena, np.full(len(X_entrena), abrev))
    # Distancia de cada texto del cluster a su centroide: ¿la frase está tan cerca como un texto típico del cluster?
    internas = metrica.get_distance(Z[etiquetas == c], centroides[c:c + 1])[:, 0]
    distancia = metrica.get_distance(z, centroides[c:c + 1])[0, 0]
    sentido, n, total = nombres[c]
    print(f"\nMÉTRICAS DEL CLUSTER {c}")
    print(f"  Tamaño: {total} textos de entrenamiento")
    print(f"  Pureza: {n}/{total} ({n / total:.0%}) son del sentido que predomina, '{sentido}'")
    print(f"  Distancia de la frase al centroide: {distancia:.3f}  (textos del cluster: media {internas.mean():.3f}, "
          f"máxima {internas.max():.3f})")
    print(f"  La frase está más cerca del centroide que el {(internas > distancia).mean():.0%} de los textos del cluster")
    if n_vecinos:  # con --paso-a-paso ya han salido en el paso 6
        cercania = metrica.get_distance(z, Z)[0]
        print(f"\n  Los {n_vecinos} textos del cluster {c} más parecidos a la frase:")
        for i in [i for i in np.argsort(cercania) if etiquetas[i] == c][:n_vecinos]:
            fila = entrena.iloc[i]
            print(f"   distancia {cercania[i]:.3f} · sentido real '{fila['sentido']}'")
            print(f"      ...{re.sub(r'\s+', ' ', fila['texto'])[:140]}...")
    if os.path.exists(ruta_por_abreviatura):
        fila = pd.read_csv(ruta_por_abreviatura).set_index("abreviatura").loc[abrev]
        print(f"\nMÉTRICAS DEL MODELO DE {abrev}")
        print(f"  k elegida (silueta): {int(fila['k'])}   k real (nº de sentidos): {int(fila['k_real'])}")
        print(f"  Silueta (entrenamiento): {fila['silueta_entrena']:.3f}")
        print(f"  En prueba (textos no vistos): ARI {fila['ARI_prueba']:.3f} · NMI {fila['NMI_prueba']:.3f} · "
              f"pureza {fila['Pureza_prueba']:.3f}")


def anotar_prueba(ruta, nombre, modelo, abrev, texto, vector, nombres):
    """Añade la prueba a un archivo común con todas las frases probadas y a qué cluster fue cada una."""
    metrica = METRICAS[modelo.metrica]()
    z = metrica.format_data(modelo._transformar(abrev, vector))
    distancias = metrica.get_distance(z, modelo.partes[abrev]["centroides"])[0]
    orden = np.argsort(distancias)
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(f"[{nombre}] {datetime.now():%d/%m/%Y %H:%M} · {abrev}\n")
        f.write(f"  Frase: {texto}\n")
        f.write(f"  -> cluster {orden[0]} (predomina '{nombres[orden[0]][0]}'), distancia {distancias[orden[0]]:.3f}\n")
        if len(orden) > 1:
            f.write(f"     segundo: cluster {orden[1]} (predomina '{nombres[orden[1]][0]}'), "
                    f"distancia {distancias[orden[1]]:.3f}\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Asignar textos nuevos a los clusters del modelo unificado")
    parser.add_argument("--abreviatura", required=True)
    parser.add_argument("--texto", nargs="+", default=None,
                        help="Una o varias frases (cada una entre comillas) que contengan la abreviatura")
    parser.add_argument("--archivo", default=None, help="En vez de --texto: un .txt con una frase por línea")
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

    carpeta = os.path.dirname(args.modelo) or "."
    frases = list(args.texto or [])
    if args.archivo:
        with open(args.archivo, encoding="utf-8") as f:
            frases += [linea.strip() for linea in f if linea.strip() and not linea.startswith("#")]  # "#" = comentario
    for frase in frases:  # que no falle a mitad: se comprueban todas antes de cargar nada
        preparar_texto(frase, abrev)

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

    def cabecera():
        print(f"\nModelo entrenado con {modelo.metodo}"
              + (f" + PCA {modelo.varianza_pca:.0%}" if modelo.varianza_pca else "") + f" ({modelo.metrica}, {modelo.init})")
        print(f"\n{abrev}: {len(nombres)} clusters ({grupo['sentido'].nunique()} sentidos reales)")
        for c, (sentido, n, total) in nombres.items():
            print(f"  cluster {c}: predomina '{sentido}' ({n} de sus {total} textos de entrenamiento)")

    # Cada frase probada se guarda en <carpeta del modelo>/demo/demo_<abreviatura>_<n>.txt (y .png)
    demo = f"{carpeta}/demo"
    for numero, frase in enumerate(frases, 1):
        os.makedirs(demo, exist_ok=True)
        nombre = siguiente_numero(abrev, demo)
        archivo = open(f"{demo}/{nombre}.txt", "w", encoding="utf-8")
        sys.stdout = Duplicar(sys.__stdout__, archivo)
        if len(frases) > 1:
            print("\n" + "#" * 70 + f"\nFRASE {numero} DE {len(frases)}\n" + "#" * 70)
        print(f"Prueba del {datetime.now():%d/%m/%Y %H:%M}")
        print(f"Abreviatura: {abrev}\nFrase: {frase}")
        cabecera()
        ventana = preparar_texto(frase, abrev)
        vector, tokens, tokenizer = vectorizar(ventana, abrev, modelo.metodo, args)
        vector = vector.astype(np.float64)
        if args.paso_a_paso:
            paso_a_paso(modelo, abrev, ventana, tokens, tokenizer, vector, entrena, X_entrena, nombres)
            from graficos_modelo_unificado import figura_frase
            print("\n7. Imagen:")
            figura_frase(modelo, abrev, entrena, X_entrena, vector, ventana, f"{demo}/{nombre}.png")
        c = int(modelo.asignar(vector, [abrev])[0])
        print(f"\nTexto: {ventana}")
        print(f"-> asignado al cluster {c} (predomina '{nombres[c][0]}')")
        resumen_cluster(modelo, abrev, c, vector, entrena, X_entrena, nombres,
                        f"{carpeta}/modelo_unificado_por_abreviatura.csv", n_vecinos=0 if args.paso_a_paso else 5)
        anotar_prueba(f"{demo}/todas_las_pruebas.txt", nombre, modelo, abrev, frase, vector, nombres)
        print(f"\nGuardado en {demo}/{nombre}.txt (y añadido a {demo}/todas_las_pruebas.txt)")
        sys.stdout = sys.__stdout__
        archivo.close()
    if not frases:
        cabecera()
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
