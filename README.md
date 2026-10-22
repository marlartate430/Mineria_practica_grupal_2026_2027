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
2. Ejecutar `comparativa_kmeans_propio.py`:
    ```bat
    python comparativa_kmeans_propio.py
    ```
3. Sabiendo ahora cuál es el mejor método de vectorización ejecutad `train.py`:
    ```bat
    python train.py --datos <ruta> --distance <distance_type> --init <init_type>
    ```
    - Donde `<ruta>` es la ruta del archivo que está en `representaciones` de extensión
        `npy` o `npz`.
    - `<distance_type>` puede ser `euclidean` o `cosine`
    - `<init_type` puede ser `plus`, `random` o `parallel`
4. Comparad el resultado de `train.py` con `buscar_mejor_k.py`

# Modelo unificado
Un único modelo que guarda dentro un K-Means por abreviatura (con su PCA y sus centroides). Cada texto se asigna
al cluster del centroide más cercano, pero solo entre los de su abreviatura. La k de cada abreviatura se elige con
la silueta (búsqueda exponencial + dicotómica de `buscar_k.py`), sin mirar los sentidos. En cada abreviatura se
aparta el 20 % de los textos de cada sentido como prueba: el modelo se entrena con el 80 % restante.
1. Entrenar (unos 3 minutos; por defecto `bert_media` + PCA 80 %, coseno, kmeans++):
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
