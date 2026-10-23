# Proyecto Minería
## Advertencia
- **NO** añadir los CSVs. 
- Los datasets estarán dentro de la carpeta `datasets`. Si no existe en vuestro proyecto lo añadís.

## Enunciado
Dados los documentos, la edad, el género y el lugar, se ha etiquetado una enfermedad. Como hay demasiadas etiquetas, dichas etiquetas se quieren agrupar en 20 capítulos

# Experimentación
1. Ejecutar `representacion_vectorial.py`:
    ```bat
    python representacion_vectorial.py --datos <ruta>.zip
    ```
    Descripción cuantitativa de los datos para el informe (instancias, repetidas, valores perdidos, vocabulario,
    OOV y rango de los atributos; unos 3 minutos). La guarda en una página con tablas, `representaciones/analisis_datos.html` (se abre con el navegador)
    y una fila por medida en `analisis_datos.csv`:
    ```bat
    python analisis_datos.py
    ```
2. Ejecutar `comparativa_kmeans_propio.py`:
    ```bat
    python comparativa_kmeans_propio.py
    ```
    Comparativa completa (todas las representaciones × sin PCA/80/90/95 % × coseno/euclídea × random/kmeans++/kmeans‖,
    180 combinaciones, unas 4 horas). Guarda cada combinación al terminarla; si se corta, se repite con `--reanudar`:
    ```bat
    python comparativa_kmeans_propio.py --varianza-pca 0.8 0.9 0.95 --metrica coseno euclidea --init kmeans++ random "kmeans||" --n-semillas 10
    ```
3. Sabiendo ahora cuál es el mejor método de vectorización ejecutad `train.py`:
    ```bat
    python train.py --datos <ruta> --distance <distance_type> --init <init_type>
    ```
    - Donde `<ruta>` es la ruta del archivo que está en `representaciones` de extensión
        `npy` o `npz`.
    - `<distance_type>` puede ser `euclidean` o `cosine`
    - `<init_type` puede ser `plus`, `random` o `parallel`
4. Comparad el resultado de `train.py` con `buscar_k.py` (elige la k de cada abreviatura con la silueta y compara
    la búsqueda dicotómica, la exponencial + dicotómica y la exhaustiva; por defecto, la ganadora de la comparativa y
    300 abreviaturas):
    ```bat
    python buscar_k.py
    ```
5. Gráficos de la comparativa (en `representaciones/graficos/`; no vuelven a ejecutar la comparativa):
    ```bat
    python graficos_comparativa.py
    python graficos_clusters.py
    ```

# Modelo unificado
Un único modelo que guarda dentro un K-Means por abreviatura (con su PCA y sus centroides). Cada texto se asigna
al cluster del centroide más cercano, pero solo entre los de su abreviatura. La k de cada abreviatura se elige con
la silueta (búsqueda exponencial + dicotómica de `buscar_k.py`), sin mirar los sentidos. En cada abreviatura se
aparta el 20 % de los textos de cada sentido como prueba: el modelo se entrena con el 80 % restante.
1. Entrenar (unos 3 minutos; por defecto la ganadora de `comparativa_kmeans_propio.csv`: `bert_media` + PCA 80 %,
   coseno, random):
    ```bat
    python modelo_unificado.py
    ```
    Guarda en `modelos_unificado/` el modelo (`modelo_unificado.npz`, unos 230 MB: **no subirlo**), un resumen con
    las métricas de entrenamiento y de prueba y una tabla por abreviatura.
2. Asignar textos nuevos a los clusters:
    ```bat
    python probar_modelo_unificado.py --abreviatura CEA --ejemplos 10
    python probar_modelo_unificado.py --abreviatura CEA --texto "serum CEA levels were elevated in colorectal cancer"
    ```
    - `--ejemplos`: textos de prueba de MeDAL (no vistos al entrenar) con su cluster y su sentido real.
    - `--texto`: una frase en inglés con la abreviatura tal cual; se vectoriza con BiomedBERT.
    - `--paso-a-paso` (con `--texto`): enseña cada paso por dentro (ventana, tokens, vector, PCA de la abreviatura,
      distancia a cada centroide y los 5 textos de entrenamiento más parecidos) y guarda una imagen con la frase
      sobre el mapa de la abreviatura en `modelos_unificado/graficos/frase_<abreviatura>_<n>.png`:
    ```bat
    python probar_modelo_unificado.py --abreviatura CEA --texto "the CEA of epidural analgesia during labour showed it was cheaper per patient" --paso-a-paso
    ```
3. Gráficos del modelo (no vuelve a entrenar), en `modelos_unificado/graficos/`:
    ```bat
    python graficos_modelo_unificado.py                      # CEA, PCA y 5-HT
    python graficos_modelo_unificado.py --abreviaturas RA MS
    ```
    - `k_elegida_vs_real.png`, `ari_por_sentidos.png` y `ari_entrena_vs_prueba.png`: el modelo entero.
    - `clusters_<abreviatura>.png`: mapa 2D de una abreviatura con sus sentidos reales y los clusters del modelo.

# Ver el K-Means paso a paso
`visualizar_kmeans.py` ejecuta el `fit()` de nuestro K-Means con los textos de una abreviatura y apunta los
centroides en cada vuelta: cómo se eligen los centroides iniciales (por defecto al azar, como la configuración ganadora), cómo cada texto va a su centroide más cercano, cómo se mueven los
centroides hasta que se paran y cómo se asigna una frase nueva. Es una demostración del algoritmo (k = nº de sentidos
y una frase de cada sentido apartada como "nueva"), no el modelo unificado.
```bat
python visualizar_kmeans.py --abreviatura CEA
python visualizar_kmeans.py --abreviatura CEA --k 3      # con otra k
python visualizar_kmeans.py --abreviatura CEA --init kmeans++   # para ver cómo elige kmeans++ los centroides
```
Guarda en `representaciones/graficos/`:
- `proceso_kmeans_<abreviatura>.png`: resumen en 6 viñetas.
- `proceso_kmeans_<abreviatura>.gif`: animación vuelta a vuelta (se abre con el visor de imágenes o el navegador).
- `proceso_kmeans_<abreviatura>.json`: los datos de cada vuelta.
