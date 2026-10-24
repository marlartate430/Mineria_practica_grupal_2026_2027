# Resultados

Copia de los resultados con los que se han hecho las tablas y figuras del informe. Cada archivo lo genera un script
del proyecto (entre paréntesis); para obtenerlos de nuevo, ver el README principal.

## Clustering

| Archivo | Contenido |
|---|---|
| `analisis_datos.csv` | Descripción del dataset (`analisis_datos.py`): instancias y atributos, repetidas, valores perdidos, vocabulario, palabras fuera de vocabulario y rango de los atributos. Una fila por medida. |
| `comparativa_kmeans_propio.csv` | Comparativa completa (`comparativa_kmeans_propio.py`): 180 combinaciones de representación × PCA × métrica × inicialización, con ARI, NMI y Pureza (y media y desviación del ARI entre semillas). La ganadora es `bert_media` + PCA 80 % + coseno + random. |
| `buscar_k_por_abreviatura.csv` | Búsqueda de k (`buscar_k.py`) en 300 abreviaturas: k real y k elegida por la búsqueda dicotómica, la exponencial + dicotómica y la exhaustiva, con el número de entrenamientos, los segundos y el ARI de cada una. |
| `modelo_unificado_por_abreviatura.csv` | Modelo final (`modelo_unificado.py`), un K-Means por abreviatura: k real, k elegida y métricas internas y externas en entrenamiento (80 %) y prueba (20 %). |
| `modelo_unificado_resumen.json` | Configuración del modelo final y la media de esas métricas sobre las 3170 abreviaturas. |

## Vectorización (`vectorizacion/`)

Generados por `representacion_vectorial.py`.

| Archivo | Contenido |
|---|---|
| `similitud_pares.csv` | Similitud coseno de pares de frases (sinónimos, negación, …) con cada representación: BoW, TF-IDF, LSA, LDA, Word2Vec, Doc2Vec, Sentence-Transformers y BERT. |
| `polisemia.csv` | Pares de frases con la misma abreviatura en el mismo o distinto sentido: similitud con BERT (vector de la abreviatura) y con Word2Vec. |
| `dispersas_dimensionalidad.csv` | Dimensiones, densidad y memoria de las representaciones dispersas según los n-gramas y `min_df`. |
| `lda_temas.csv` | Palabras más representativas de cada tema de LDA. |
| `word2vec_analogias.csv` | Analogías resueltas con Word2Vec. |

No se incluyen `instancias.csv` (87 MB, son los datos de entrada) ni `comparativa_kmeans_propio_por_abreviatura.csv`
(39 MB, el detalle por abreviatura que resume `comparativa_kmeans_propio.csv`).
