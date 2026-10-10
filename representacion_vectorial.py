"""
Representación vectorial de MeDAL (Medical Abbreviation Disambiguation) y comparativa de técnicas.

Tarea: para cada abreviatura ambigua (p. ej. CS, CA, RT) se representan sus contextos y se
comprueba qué representación separa mejor sus significados (columna LABEL). Cada instancia es
una ventana de palabras alrededor de la abreviatura dentro de un abstract de PubMed.

Técnicas:
    Dispersas    : bow, tfidf, lsa (TF-IDF + TruncatedSVD)
    Temas        : lda (Latent Dirichlet Allocation)
    Estáticas    : w2v (word2vec entrenado con abstracts de MeDAL), d2v (doc2vec)
    Contextuales : st (sentence-transformers), bert (BiomedBERT: vector de la abreviatura
                   en su contexto -> 'bert_abrev', y media de la ventana -> 'bert_media')

Uso:
    python representacion_vectorial.py --datos ~/Escritorio/MeDAL/data/full_data.csv.zip
    python representacion_vectorial.py --metodos tfidf w2v bert --abreviaturas CS CA RT
    python representacion_vectorial.py --n-filas 50000 --n-abreviaturas 4      # prueba rápida

Resultados en --salida (por defecto 'representaciones/'):
    instancias.csv, X_<metodo>.npz/.npy, comparativa.csv, comparativa_por_abreviatura.csv,
    dispersas_dimensionalidad.csv, similitud_pares.csv, polisemia.csv, mapa_2d.png,
    word2vec_palabras.png, word2vec_analogias.csv, projector/ (TensorFlow Projector)
"""
import argparse
import csv
import io
import os
import random
import re
import struct
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

TODOS_METODOS = ["bow", "tfidf", "lsa", "lda", "w2v", "d2v", "st", "bert"]

RUTAS_DATOS = [
    "datasets/full_data.csv",
    "datasets/full_data.csv.zip",
    os.path.expanduser("~/Escritorio/MeDAL/data/full_data.csv"),
    os.path.expanduser("~/Escritorio/MeDAL/data/full_data.csv.zip"),
    os.path.expanduser("~/Mineria de Datos/MeDAL/programa.deb"),
]

# La negación cambia el sentido clínico ("no evidence of tumor"): no se elimina
STOPWORDS = sorted(ENGLISH_STOP_WORDS - {"no", "not", "nor", "without"})

PARES_FRASES = [
    ("sinonimos", "the patient suffered a myocardial infarction", "the patient had a heart attack"),
    ("negacion", "the tumor responded to treatment", "the tumor did not respond to treatment"),
    ("polisemia", "patients with MS showed demyelinating lesions in the brain",
     "the peptides were identified by MS after tryptic digestion"),
    ("no_relacionadas", "insulin resistance in obese patients", "the bacterial strain was grown in broth"),
]

# MS = multiple sclerosis / mass spectrometry
FRASES_POLISEMIA = [
    ("multiple sclerosis", "patients with MS showed demyelinating lesions in the white matter"),
    ("multiple sclerosis", "relapses of MS were treated with interferon beta"),
    ("mass spectrometry", "the peptides were identified by MS after tryptic digestion"),
    ("mass spectrometry", "metabolites were quantified using liquid chromatography coupled to MS"),
]

PALABRAS_CLAVE_W2V = ["cancer", "insulin", "heart", "liver", "infection", "blood", "brain", "dose", "children",
                      "antibody"]

# a : b :: c : ?   ->   most_similar(positive=[b, c], negative=[a])
ANALOGIAS = [("kidney", "renal", "liver"), ("heart", "cardiac", "lung"),
             ("children", "pediatric", "elderly")]


# ---------------------------------------------------------------------------
# Lectura de MeDAL
# ---------------------------------------------------------------------------

def _lineas_zip_incompleto(ruta):
    """Descomprime en streaming la primera entrada de un zip aunque la descarga esté cortada."""
    with open(ruta, "rb") as f:
        cabecera = f.read(30)
        if cabecera[:4] != b"PK\x03\x04":
            raise SystemExit(f"{ruta} no es un zip ni un csv")
        long_nombre, long_extra = struct.unpack("<HH", cabecera[26:30])
        f.read(long_nombre + long_extra)
        descompresor = zlib.decompressobj(-zlib.MAX_WBITS)
        resto = b""
        while not descompresor.eof:
            bloque = f.read(1 << 20)
            if not bloque:
                break
            try:
                resto += descompresor.decompress(bloque)
            except zlib.error:
                break
            *lineas, resto = resto.split(b"\n")
            for linea in lineas:
                yield linea.decode("utf-8", "replace")
        # Si la descarga está cortada, la última línea está incompleta y se descarta
        if descompresor.eof and resto:
            yield resto.decode("utf-8", "replace")


_AVISADOS = set()


def _lineas(ruta):
    if ruta.endswith(".csv"):
        with open(ruta, encoding="utf-8", newline="") as f:
            yield from f
        return
    try:
        with zipfile.ZipFile(ruta) as zf:
            nombre = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(nombre) as f:
                yield from io.TextIOWrapper(f, encoding="utf-8", newline="")
    except zipfile.BadZipFile:
        if ruta not in _AVISADOS:
            _AVISADOS.add(ruta)
            print(f"  Aviso: {ruta} está incompleto; se leen las filas que contiene")
        yield from _lineas_zip_incompleto(ruta)


def leer_filas(ruta, n_filas):
    """Genera (texto, [(posición, etiqueta), ...]) de las primeras n_filas de MeDAL."""
    lector = csv.DictReader(_lineas(ruta))
    for i, fila in enumerate(lector):
        if i >= n_filas:
            return
        try:
            posiciones = [int(p) for p in fila["LOCATION"].split("|")]
            etiquetas = fila["LABEL"].split("|")
        except (KeyError, ValueError, AttributeError):
            continue
        yield fila["TEXT"], list(zip(posiciones, etiquetas))


def es_abreviatura(token):
    # Descarta falsos positivos de MeDAL como 'one', 'exp' o 'gl'
    return any(c.isupper() for c in token)


def ventana(palabras, i, w):
    """Texto de la ventana ±w palabras alrededor de la posición i y posición (en caracteres) de la abreviatura."""
    izquierda = palabras[max(0, i - w):i]
    texto_izq = " ".join(izquierda)
    inicio = len(texto_izq) + (1 if izquierda else 0)
    texto = " ".join(izquierda + [palabras[i]] + palabras[i + 1:i + 1 + w])
    return texto, inicio, inicio + len(palabras[i])


def construir_instancias(args):
    """
    Pasada 1: cuenta abreviaturas y sentidos. Elige abreviaturas con al menos 2 sentidos frecuentes.
    Pasada 2: muestreo por reservorio de hasta max_por_sentido contextos de cada (abreviatura, sentido).
    """
    conteo = defaultdict(Counter)
    for texto, pares in leer_filas(args.datos, args.n_filas):
        palabras = texto.split()
        for pos, etiqueta in pares:
            if 0 <= pos < len(palabras) and es_abreviatura(palabras[pos]):
                conteo[palabras[pos]][etiqueta] += 1

    sentidos = {}
    candidatas = args.abreviaturas or list(conteo)
    for abrev in candidatas:
        frecuentes = [(e, c) for e, c in conteo[abrev].most_common(args.max_sentidos) if c >= args.min_por_sentido]
        if len(frecuentes) >= 2:
            sentidos[abrev] = frecuentes
    if args.abreviaturas:
        descartadas = set(args.abreviaturas) - set(sentidos)
        if descartadas:
            print(f"  Sin al menos 2 sentidos con {args.min_por_sentido}+ ejemplos: {sorted(descartadas)}")
    else:
        # Las más ambiguas: mayor número de ejemplos en su segundo sentido más frecuente
        elegidas = sorted(sentidos, key=lambda a: -sentidos[a][1][1])[:args.n_abreviaturas]
        sentidos = {a: sentidos[a] for a in elegidas}
    if not sentidos:
        raise SystemExit("Ninguna abreviatura cumple los requisitos: sube --n-filas o baja --min-por-sentido")

    objetivo = {(a, e) for a, lista in sentidos.items() for e, _ in lista}
    rng = random.Random(args.semilla)
    reservorio, vistos = defaultdict(list), Counter()
    for doc, (texto, pares) in enumerate(leer_filas(args.datos, args.n_filas)):
        palabras = texto.split()
        for pos, etiqueta in pares:
            if not 0 <= pos < len(palabras) or (palabras[pos], etiqueta) not in objetivo:
                continue
            clave = (palabras[pos], etiqueta)
            vistos[clave] += 1
            j = vistos[clave] - 1 if len(reservorio[clave]) < args.max_por_sentido else rng.randrange(vistos[clave])
            if j < args.max_por_sentido:
                instancia = (doc, pos, *ventana(palabras, pos, args.ventana))
                if len(reservorio[clave]) < args.max_por_sentido:
                    reservorio[clave].append(instancia)
                else:
                    reservorio[clave][j] = instancia

    filas = [{"abreviatura": a, "sentido": e, "doc": d, "posicion": p, "texto": t, "ini": i, "fin": f}
             for (a, e), lista in reservorio.items() for d, p, t, i, f in lista]
    return pd.DataFrame(filas).sort_values(["abreviatura", "sentido", "doc"]).reset_index(drop=True)


class CorpusMedal:
    """Iterable reiniciable de abstracts tokenizados (gensim lo recorre una vez por época)."""

    def __init__(self, ruta, n_filas, etiquetado=False):
        self.ruta, self.n_filas, self.etiquetado = ruta, n_filas, etiquetado

    def __iter__(self):
        from gensim.models.doc2vec import TaggedDocument
        for i, (texto, _) in enumerate(leer_filas(self.ruta, self.n_filas)):
            tokens = tokenizar(texto)
            yield TaggedDocument(tokens, [i]) if self.etiquetado else tokens


# ---------------------------------------------------------------------------
# Preprocesado
# ---------------------------------------------------------------------------

def tokenizar(texto):
    # MeDAL ya viene sin puntuación ni números; solo se pasa a minúsculas
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", texto.lower())


# ---------------------------------------------------------------------------
# Representaciones
# ---------------------------------------------------------------------------

class Representacion:
    nombre = ""
    tipo = ""

    def fit_transform(self, textos):
        raise NotImplementedError

    def transform(self, textos):
        raise NotImplementedError


class BolsaPalabras(Representacion):
    nombre, tipo = "bow", "dispersa"

    def __init__(self, min_df=2, ngramas=(1, 1)):
        self.vectorizer = CountVectorizer(stop_words=STOPWORDS, ngram_range=ngramas, min_df=min_df)

    def fit_transform(self, textos):
        return self.vectorizer.fit_transform(textos)

    def transform(self, textos):
        return self.vectorizer.transform(textos)


class TFIDF(BolsaPalabras):
    nombre, tipo = "tfidf", "dispersa"

    def __init__(self, min_df=2, ngramas=(1, 2)):
        self.vectorizer = TfidfVectorizer(stop_words=STOPWORDS, ngram_range=ngramas, min_df=min_df,
                                          max_df=0.9, sublinear_tf=True)


class LSA(Representacion):
    """TF-IDF reducido con TruncatedSVD (cambio de base): combate la dimensionalidad."""
    nombre, tipo = "lsa", "densa (reducción de dispersa)"

    def __init__(self, min_df=2, n_componentes=300):
        self.tfidf = TFIDF(min_df=min_df)
        self.n_componentes = n_componentes

    def fit_transform(self, textos):
        X = self.tfidf.fit_transform(textos)
        n = max(2, min(self.n_componentes, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n, random_state=42)
        X_red = self.svd.fit_transform(X)
        print(f"  LSA: {n} componentes explican {self.svd.explained_variance_ratio_.sum():.1%} de la varianza")
        return X_red

    def transform(self, textos):
        return self.svd.transform(self.tfidf.transform(textos))


class LDA(Representacion):
    """Topic modeling: cada texto es una distribución de probabilidad sobre n temas."""
    nombre, tipo = "lda", "densa (temas)"

    def __init__(self, min_df=2, n_temas=20):
        self.bow = BolsaPalabras(min_df=min_df)
        self.lda = LatentDirichletAllocation(n_components=n_temas, learning_method="batch", random_state=42)

    def fit_transform(self, textos):
        return self.lda.fit_transform(self.bow.fit_transform(textos))

    def transform(self, textos):
        return self.lda.transform(self.bow.transform(textos))

    def temas(self, n_palabras=8):
        vocab = self.bow.vectorizer.get_feature_names_out()
        return [" ".join(vocab[j] for j in fila.argsort()[::-1][:n_palabras]) for fila in self.lda.components_]


class Word2VecMedia(Representacion):
    """Vector de la ventana = media de los vectores word2vec de sus palabras (estático)."""
    nombre, tipo = "w2v", "densa estática"

    def __init__(self, ruta, n_filas, dim=200, preentrenado=None):
        self.ruta, self.n_filas, self.dim, self.preentrenado = ruta, n_filas, dim, preentrenado

    def fit_transform(self, textos):
        from gensim.models import KeyedVectors, Word2Vec
        if self.preentrenado:
            print(f"  Cargando vectores preentrenados de {self.preentrenado}...")
            self.kv = KeyedVectors.load_word2vec_format(self.preentrenado, binary=self.preentrenado.endswith(".bin"),
                                                        limit=1_000_000)
        else:
            print(f"  Entrenando word2vec (skip-gram) con {self.n_filas} abstracts...")
            modelo = Word2Vec(sentences=CorpusMedal(self.ruta, self.n_filas), vector_size=self.dim, window=5,
                              min_count=5, sg=1, epochs=5, workers=os.cpu_count() or 4, seed=42)
            self.kv = modelo.wv
        print(f"  Vocabulario: {len(self.kv.key_to_index)} palabras")
        return self.transform(textos)

    def transform(self, textos):
        X = np.zeros((len(textos), self.kv.vector_size), dtype=np.float32)
        for i, texto in enumerate(textos):
            vectores = [self.kv[w] for w in tokenizar(texto) if w in self.kv.key_to_index]
            if vectores:
                X[i] = np.mean(vectores, axis=0)
        return X


class Doc2VecRep(Representacion):
    nombre, tipo = "d2v", "densa estática"

    def __init__(self, ruta, n_filas, dim=100):
        self.ruta, self.n_filas, self.dim = ruta, n_filas, dim

    def fit_transform(self, textos):
        from gensim.models.doc2vec import Doc2Vec
        print(f"  Entrenando doc2vec (PV-DBOW) con {self.n_filas} abstracts...")
        self.modelo = Doc2Vec(CorpusMedal(self.ruta, self.n_filas, etiquetado=True), vector_size=self.dim, dm=0,
                              min_count=5, epochs=10, workers=os.cpu_count() or 4, seed=42)
        return self.transform(textos)

    def transform(self, textos):
        return np.vstack([self.modelo.infer_vector(tokenizar(t)) for t in textos])


class SentenceTransformerRep(Representacion):
    """Transformer ajustado para que la similitud coseno refleje similitud semántica."""
    nombre, tipo = "st", "densa contextual"

    def __init__(self, modelo, batch_size=32):
        from sentence_transformers import SentenceTransformer
        self.modelo = SentenceTransformer(modelo)
        self.batch_size = batch_size

    def fit_transform(self, textos):
        return self.transform(textos)

    def transform(self, textos):
        return self.modelo.encode(list(textos), batch_size=self.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=len(textos) > 500)


class BertContextual(Representacion):
    """
    BERT biomédico. De una misma pasada se obtienen:
      - bert_abrev: vector contextual de la abreviatura (media de sus subtokens)
      - bert_media: mean pooling de toda la ventana
    """
    nombre, tipo = "bert", "densa contextual"

    def __init__(self, modelo, batch_size=16, max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        if not self.tokenizer.is_fast:
            raise ImportError("se necesita un tokenizador rápido (offsets) para localizar la abreviatura")
        self.modelo = AutoModel.from_pretrained(modelo).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo.to(self.device)
        self.batch_size, self.max_length = batch_size, max_length

    def _pasada(self, textos, spans=None):
        torch = self.torch
        medias, abrevs, sin_abrev = [], [], 0
        for i in range(0, len(textos), self.batch_size):
            if len(textos) > 500 and (i // self.batch_size) % 50 == 0:
                print(f"  BERT: {i}/{len(textos)}")
            entradas = self.tokenizer(textos[i:i + self.batch_size], padding=True, truncation=True,
                                      max_length=self.max_length, return_offsets_mapping=True, return_tensors="pt")
            offsets = entradas.pop("offset_mapping")
            entradas = {k: v.to(self.device) for k, v in entradas.items()}
            with torch.no_grad():
                ocultos = self.modelo(**entradas).last_hidden_state
            mascara = entradas["attention_mask"].unsqueeze(-1).float()
            media = (ocultos * mascara).sum(1) / mascara.sum(1).clamp(min=1e-9)
            medias.append(media.cpu().numpy())
            if spans is not None:
                lote = spans[i:i + self.batch_size]
                ini = torch.tensor([s[0] for s in lote])[:, None]
                fin = torch.tensor([s[1] for s in lote])[:, None]
                a, b = offsets[..., 0], offsets[..., 1]
                # Subtokens cuyo rango de caracteres se solapa con la abreviatura
                sel = ((b > a) & (a < fin) & (b > ini)).unsqueeze(-1).float().to(self.device)
                cuenta = sel.sum(1)
                vector = (ocultos * sel).sum(1) / cuenta.clamp(min=1)
                # Si la abreviatura quedó truncada se usa la media de la ventana
                sin_abrev += int((cuenta == 0).sum())
                abrevs.append(torch.where(cuenta > 0, vector, media).cpu().numpy())
        if sin_abrev:
            print(f"  Aviso: {sin_abrev} instancias sin la abreviatura tras truncar (se usa la media)")
        return np.vstack(medias), (np.vstack(abrevs) if spans is not None else None)

    def representar(self, textos, spans):
        media, abrev = self._pasada(list(textos), list(spans))
        return {"bert_abrev": abrev, "bert_media": media}

    def transform(self, textos):
        return self._pasada(list(textos))[0]

    def vector_palabra(self, frase, palabra):
        inicio = frase.find(palabra)
        return None if inicio < 0 else self._pasada([frase], [(inicio, inicio + len(palabra))])[1][0]


def crear_representacion(metodo, args):
    if metodo == "bow":
        return BolsaPalabras(min_df=args.min_df)
    if metodo == "tfidf":
        return TFIDF(min_df=args.min_df)
    if metodo == "lsa":
        return LSA(min_df=args.min_df, n_componentes=args.n_lsa)
    if metodo == "lda":
        return LDA(min_df=args.min_df, n_temas=args.n_temas)
    if metodo == "w2v":
        return Word2VecMedia(args.datos, args.n_textos_w2v, preentrenado=args.w2v_preentrenado)
    if metodo == "d2v":
        return Doc2VecRep(args.datos, args.n_textos_w2v)
    if metodo == "st":
        return SentenceTransformerRep(args.modelo_st)
    if metodo == "bert":
        return BertContextual(args.modelo_bert)
    raise ValueError(metodo)


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def analizar_dimensionalidad(textos, min_df):
    """Cómo crece el vocabulario (= dimensiones) de TF-IDF con los n-gramas."""
    filas = []
    for ngramas in [(1, 1), (1, 2), (1, 3)]:
        for df_min in sorted({1, min_df}):
            X = TfidfVectorizer(stop_words=STOPWORDS, ngram_range=ngramas, min_df=df_min).fit_transform(textos)
            filas.append({
                "ngramas": f"{ngramas[0]}-{ngramas[1]}", "min_df": df_min,
                "dimensiones": X.shape[1],
                "densidad": X.nnz / (X.shape[0] * X.shape[1]),
                "memoria_dispersa_MB": (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6,
                "memoria_si_fuera_densa_MB": X.shape[0] * X.shape[1] * 8 / 1e6,
            })
    return pd.DataFrame(filas)


def coseno(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def evaluar(X, y, semilla=42):
    """Métricas de una abreviatura: cuánto separa la representación sus sentidos."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = np.asarray(y)
    clases, cuentas = np.unique(y, return_counts=True)
    # Norma 1: la distancia euclídea pasa a ser equivalente a la distancia coseno
    Xn = normalize(X)
    km = KMeans(n_clusters=len(clases), n_init=10, random_state=semilla).fit(Xn)
    S = Xn @ Xn.T
    S = S.toarray() if sparse.issparse(S) else S
    mismo = y[:, None] == y[None, :]
    np.fill_diagonal(mismo, False)
    distinto = y[:, None] != y[None, :]
    resultado = {
        "ARI": adjusted_rand_score(y, km.labels_),
        "NMI": normalized_mutual_info_score(y, km.labels_),
        "silueta": silhouette_score(Xn, y, metric="cosine", random_state=semilla),
        "sim_intra": S[mismo].mean(),
        "sim_inter": S[distinto].mean(),
    }
    resultado["separacion"] = resultado["sim_intra"] - resultado["sim_inter"]
    folds = min(5, cuentas.min())
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semilla)
        resultado["f1_sonda"] = cross_val_score(LogisticRegression(max_iter=2000), Xn, y, cv=cv,
                                                scoring="f1_macro").mean()
    return resultado


def similitud_pares(representaciones):
    filas = []
    for etiqueta, f1, f2 in PARES_FRASES:
        fila = {"tipo": etiqueta, "frase_1": f1, "frase_2": f2}
        for nombre, rep in representaciones.items():
            V = rep.transform([f1, f2])
            V = V.toarray() if sparse.issparse(V) else V
            fila[nombre if nombre != "bert" else "bert_media"] = round(coseno(V[0], V[1]), 3)
        filas.append(fila)
    return pd.DataFrame(filas)


def demo_polisemia(bert, w2v=None):
    """'MS' en contextos distintos: un vector por palabra (estático) vs. uno por contexto."""
    vectores = [bert.vector_palabra(frase, "MS") for _, frase in FRASES_POLISEMIA]
    filas = []
    for i in range(len(FRASES_POLISEMIA)):
        for j in range(i + 1, len(FRASES_POLISEMIA)):
            (s1, f1), (s2, f2) = FRASES_POLISEMIA[i], FRASES_POLISEMIA[j]
            fila = {"frase_a": f1, "frase_b": f2, "mismo_sentido": s1 == s2,
                    "bert_abrev": round(coseno(vectores[i], vectores[j]), 3)}
            if w2v is not None:
                # word2vec asigna el mismo vector a 'ms' en todas las frases
                fila["w2v"] = 1.0 if "ms" in w2v.kv.key_to_index else None
            filas.append(fila)
    return pd.DataFrame(filas)


def proyectar_2d(X, metodo, semilla=42):
    n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
    if X.shape[1] > 50:
        X = TruncatedSVD(n_components=n_comp, random_state=semilla).fit_transform(normalize(X))
    else:
        X = normalize(X.toarray() if sparse.issparse(X) else X)
    if metodo == "tsne":
        from sklearn.manifold import TSNE
        perplejidad = max(2, min(30, (X.shape[0] - 1) // 3))
        return TSNE(n_components=2, perplexity=perplejidad, init="pca", random_state=semilla).fit_transform(X)
    return PCA(n_components=2, random_state=semilla).fit_transform(X)


def dibujar_mapas(matrices, y, titulo, ruta, metodo_2d, semilla=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    clases = sorted(set(y))
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    n = len(matrices)
    columnas = min(3, n)
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4.5 * filas), squeeze=False)
    for eje, (nombre, X) in zip(ejes.ravel(), matrices.items()):
        P = proyectar_2d(X, metodo_2d, semilla)
        for k, clase in enumerate(clases):
            m = y == clase
            eje.scatter(P[m, 0], P[m, 1], s=8, alpha=0.6, color=colores[k % 10], label=str(clase))
        eje.set_title(f"{nombre} ({X.shape[1]} dims)")
        eje.set_xticks([])
        eje.set_yticks([])
    for eje in ejes.ravel()[n:]:
        eje.axis("off")
    ejes.ravel()[0].legend(markerscale=2, fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def dibujar_word2vec(w2v, ruta, topn=6):
    """Palabras clave del dominio y sus vecinos más cercanos en el espacio word2vec."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    semillas = [p for p in PALABRAS_CLAVE_W2V if p in w2v.kv.key_to_index]
    if not semillas:
        return {}
    palabras, grupos, vecinos = [], [], {}
    for g, p in enumerate(semillas):
        similares = w2v.kv.most_similar(p, topn=topn)
        vecinos[p] = [v for v, _ in similares]
        for w in [p] + vecinos[p]:
            if w not in palabras:
                palabras.append(w)
                grupos.append(g)
    P = PCA(n_components=2, random_state=42).fit_transform(np.vstack([w2v.kv[w] for w in palabras]))
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, eje = plt.subplots(figsize=(11, 9))
    for (x, yy), w, g in zip(P, palabras, grupos):
        es_semilla = w in semillas
        eje.scatter(x, yy, color=colores[g % 10], s=60 if es_semilla else 20)
        eje.annotate(w, (x, yy), fontsize=11 if es_semilla else 8, fontweight="bold" if es_semilla else "normal",
                     xytext=(3, 3), textcoords="offset points")
    eje.set_title("word2vec (MeDAL): palabras clave y sus vecinos más cercanos (PCA)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return vecinos


def analogias_word2vec(w2v):
    filas = []
    for a, b, c in ANALOGIAS:
        if all(p in w2v.kv.key_to_index for p in (a, b, c)):
            resultado = w2v.kv.most_similar(positive=[b, c], negative=[a], topn=3)
            filas.append({"analogia": f"{a} : {b} :: {c} : ?", "respuesta": ", ".join(w for w, _ in resultado)})
    return pd.DataFrame(filas)


def exportar_projector(directorio, vectores, metadatos, cabecera=None):
    """Ficheros para https://projector.tensorflow.org (Load -> vectors.tsv y metadata.tsv)."""
    os.makedirs(directorio, exist_ok=True)
    np.savetxt(f"{directorio}/vectors.tsv", vectores, delimiter="\t", fmt="%.5f")
    with open(f"{directorio}/metadata.tsv", "w", encoding="utf-8") as f:
        if cabecera:
            f.write("\t".join(cabecera) + "\n")
        for fila in metadatos:
            f.write("\t".join(str(c).replace("\t", " ") for c in fila) + "\n")


def guardar_matriz(X, ruta_base):
    if sparse.issparse(X):
        sparse.save_npz(f"{ruta_base}.npz", X.tocsr())
    else:
        np.save(f"{ruta_base}.npy", X)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Representación vectorial de MeDAL y comparativa de técnicas")
    parser.add_argument("--datos", default=None, help="full_data.csv, .zip (aunque esté incompleto) o train.csv")
    parser.add_argument("--metodos", nargs="+", default=TODOS_METODOS, choices=TODOS_METODOS)
    parser.add_argument("--n-filas", type=int, default=300_000, help="Filas de MeDAL que se recorren")
    parser.add_argument("--abreviaturas", nargs="+", default=None, help="Abreviaturas concretas (p. ej. CS CA RT)")
    parser.add_argument("--n-abreviaturas", type=int, default=8)
    parser.add_argument("--max-sentidos", type=int, default=4, help="Sentidos más frecuentes por abreviatura")
    parser.add_argument("--min-por-sentido", type=int, default=50)
    parser.add_argument("--max-por-sentido", type=int, default=150)
    parser.add_argument("--ventana", type=int, default=25, help="Palabras a cada lado de la abreviatura")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--n-lsa", type=int, default=300)
    parser.add_argument("--n-temas", type=int, default=20)
    parser.add_argument("--n-pca", type=int, default=50,
                        help="Dimensiones de las variantes '+pca' de los transformers (0 = no calcularlas)")
    parser.add_argument("--n-textos-w2v", type=int, default=100_000, help="Abstracts para entrenar word2vec/doc2vec")
    parser.add_argument("--w2v-preentrenado", default=None, help="Fichero .vec/.bin (p. ej. BioWordVec)")
    parser.add_argument("--modelo-st", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--modelo-bert", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--abrev-mapa", default=None, help="Abreviatura que se dibuja en mapa_2d.png")
    parser.add_argument("--mapa", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--salida", default="representaciones")
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    if args.datos is None:
        existentes = [r for r in RUTAS_DATOS if os.path.exists(r)]
        if not existentes:
            raise SystemExit("No se encuentra MeDAL: indica la ruta con --datos")
        args.datos = max(existentes, key=os.path.getsize)
    print(f"Datos: {args.datos}")
    os.makedirs(args.salida, exist_ok=True)

    # 1. Instancias: contextos de abreviaturas ambiguas
    inicio = time.time()
    df = construir_instancias(args)
    textos = df["texto"].tolist()
    print(f"{len(df)} instancias de {df['abreviatura'].nunique()} abreviaturas "
          f"({time.time() - inicio:.0f} s)")
    resumen = df.groupby(["abreviatura", "sentido"]).size().rename("n").reset_index()
    print(resumen.to_string(index=False), "\n")
    df.to_csv(f"{args.salida}/instancias.csv", index=False)

    # Las abreviaturas elegidas aparecen en todas sus ventanas: no aportan para separar sentidos y
    # acaparan los temas de LDA, así que se excluyen del vocabulario de BoW/TF-IDF/LDA
    global STOPWORDS
    STOPWORDS = sorted(set(STOPWORDS) | {a.lower() for a in df["abreviatura"].unique()})

    # 2. Representaciones
    representaciones, matrices, info = {}, {}, {}
    for metodo in args.metodos:
        print(f"[{metodo}] calculando...")
        inicio = time.time()
        try:
            rep = crear_representacion(metodo, args)
            if metodo == "bert":
                resultado = rep.representar(textos, list(zip(df["ini"], df["fin"])))
            else:
                resultado = {metodo: rep.fit_transform(textos)}
        except ImportError as e:
            print(f"  Se omite '{metodo}': falta una dependencia ({e})")
            continue
        except OSError as e:
            print(f"  Se omite '{metodo}': no se pudo cargar el modelo ({e})")
            continue
        representaciones[metodo] = rep
        segundos = round(time.time() - inicio, 1)
        for nombre, X in resultado.items():
            matrices[nombre] = X
            info[nombre] = {"tipo": rep.tipo, "tiempo_s": segundos}
            print(f"  {nombre}: forma {X.shape} en {segundos} s")
    if not matrices:
        raise SystemExit("No se ha podido calcular ninguna representación")
    for nombre, X in matrices.items():
        guardar_matriz(X, f"{args.salida}/X_{nombre}")

    if "lda" in representaciones:
        temas = representaciones["lda"].temas()
        pd.DataFrame({"tema": range(len(temas)), "palabras": temas}).to_csv(f"{args.salida}/lda_temas.csv",
                                                                            index=False)

    # 3. Limitaciones de las dispersas: dimensionalidad
    if {"bow", "tfidf", "lsa"} & set(representaciones):
        dims = analizar_dimensionalidad(textos, args.min_df)
        dims.to_csv(f"{args.salida}/dispersas_dimensionalidad.csv", index=False)
        print("\nDimensionalidad de TF-IDF según n-gramas:")
        print(dims.to_string(index=False))

    # 4. Comparativa: métricas por abreviatura y media.
    # Los vectores de los transformers son anisótropos (todos apuntan en una dirección parecida y el
    # coseno entre cualquier par es alto). Centrarlos y reducirlos con PCA dentro de cada abreviatura
    # elimina esa componente común: variantes '<metodo>+pca'.
    variantes = {nombre: nombre for nombre in matrices}
    for nombre in ("bert_abrev", "bert_media", "st"):
        if nombre in matrices and args.n_pca:
            variantes[f"{nombre}+pca"] = nombre
            info[f"{nombre}+pca"] = {**info[nombre], "tipo": info[nombre]["tipo"] + " + centrado/PCA"}
    filas = []
    for nombre, origen in variantes.items():
        for abrev, grupo in df.groupby("abreviatura"):
            idx = grupo.index.to_numpy()
            X = matrices[origen][idx]
            if nombre != origen:
                X = PCA(n_components=min(args.n_pca, *X.shape), random_state=args.semilla).fit_transform(X)
            filas.append({"metodo": nombre, "abreviatura": abrev, "n_sentidos": grupo["sentido"].nunique(),
                          **evaluar(X, grupo["sentido"].values, args.semilla)})
    por_abrev = pd.DataFrame(filas)
    por_abrev.round(4).to_csv(f"{args.salida}/comparativa_por_abreviatura.csv", index=False)
    metricas = [c for c in ["ARI", "NMI", "silueta", "separacion", "f1_sonda"] if c in por_abrev]
    comparativa = por_abrev.groupby("metodo", sort=False)[metricas].mean().round(4)
    comparativa.insert(0, "dims", [args.n_pca if m.endswith("+pca") else matrices[m].shape[1]
                                   for m in comparativa.index])
    comparativa.insert(0, "tiempo_s", [info[m]["tiempo_s"] for m in comparativa.index])
    comparativa.insert(0, "tipo", [info[m]["tipo"] for m in comparativa.index])
    comparativa = comparativa.sort_values("ARI", ascending=False)
    comparativa.to_csv(f"{args.salida}/comparativa.csv")
    print("\nComparativa (media sobre abreviaturas; ARI/NMI de K-Means con k = nº de sentidos):")
    print(comparativa.to_string())

    # 5. Similitud entre pares de frases
    pares = similitud_pares(representaciones)
    pares.to_csv(f"{args.salida}/similitud_pares.csv", index=False)
    print("\nSimilitud coseno entre pares de frases:")
    print(pares.drop(columns=["frase_1", "frase_2"]).to_string(index=False))

    # 6. Estático vs. contextual: 'MS' en contextos distintos
    if "bert" in representaciones:
        polisemia = demo_polisemia(representaciones["bert"], representaciones.get("w2v"))
        polisemia.to_csv(f"{args.salida}/polisemia.csv", index=False)
        print("\nPolisemia de 'MS' (multiple sclerosis / mass spectrometry):")
        print(polisemia.drop(columns=["frase_a", "frase_b"]).assign(
            par=[f"{a[:28]}... / {b[:28]}..." for a, b in zip(polisemia["frase_a"], polisemia["frase_b"])]
        ).to_string(index=False))

    # 7. word2vec: vecinos, analogías y exportación para TensorFlow Projector
    if "w2v" in representaciones:
        w2v = representaciones["w2v"]
        vecinos = dibujar_word2vec(w2v, f"{args.salida}/word2vec_palabras.png")
        print("\nVecinos más cercanos en word2vec:")
        for palabra, lista in vecinos.items():
            print(f"  {palabra}: {', '.join(lista)}")
        analogias = analogias_word2vec(w2v)
        if not analogias.empty:
            analogias.to_csv(f"{args.salida}/word2vec_analogias.csv", index=False)
            print("\nAnalogías (a : b :: c : ?):")
            print(analogias.to_string(index=False))
        palabras = w2v.kv.index_to_key[:5000]
        exportar_projector(f"{args.salida}/projector/word2vec", w2v.kv[palabras], [[p] for p in palabras])

    # 8. Mapas 2D de una abreviatura, coloreados por sentido
    abrev_mapa = args.abrev_mapa if args.abrev_mapa in set(df["abreviatura"]) else \
        df["abreviatura"].value_counts().index[0]
    idx = df.index[df["abreviatura"] == abrev_mapa].to_numpy()
    dibujar_mapas({n: X[idx] for n, X in matrices.items()}, df.loc[idx, "sentido"].values,
                  f"Contextos de '{abrev_mapa}' coloreados por sentido ({args.mapa.upper()})",
                  f"{args.salida}/mapa_2d.png", args.mapa, args.semilla)
    mejor = next((m for m in ["bert_abrev", "st", "bert_media"] if m in matrices), None)
    if mejor:
        X = matrices[mejor][idx]
        exportar_projector(f"{args.salida}/projector/{mejor}_{abrev_mapa}",
                           X.toarray() if sparse.issparse(X) else X,
                           df.loc[idx, ["sentido", "texto"]].values.tolist(), cabecera=["sentido", "texto"])
    print(f"\nResultados guardados en '{args.salida}/'")


if __name__ == "__main__":
    sys.exit(main())
