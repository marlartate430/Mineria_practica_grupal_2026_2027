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
