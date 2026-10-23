# Versión "todos juntos" de train.py: un único K-Means para todos los textos con una sola K (elegida con el
# codo). train.py entrena un modelo por abreviatura. Resultados en modelos_finales_todos/
import numpy as np
import pandas as pd
from scipy import sparse
import os
import argparse
from kmeans_init import kmeans_plus_plus, random_cluster_selection, KMeans_parallel

from modeloKMeans import SphericalKMeans, EuclideanKMeans
from kmeans_init import kmeans_plus_plus
from comparativa_kmeans_propio import ari_por_abreviatura, pca
from evaluacion import evaluar_clustering, nmi

def kneedle_codo(k_valores, inercias):
    """Calcula el codo analíticamente usando la geometría de Kneedle."""
    k_valores = np.array(k_valores)
    inercias = np.array(inercias)
    
    # 1. Normalizar X e Y al cuadrado unitario [0, 1]
    x_norm = (k_valores - np.min(k_valores)) / (np.max(k_valores) - np.min(k_valores) + 1e-10)
    y_norm = (inercias - np.min(inercias)) / (np.max(inercias) - np.min(inercias) + 1e-10)
    
    # 2. Curva de diferencias respecto a la diagonal y = 1 - x
    diferencias = (1 - x_norm) - y_norm
    
    # 3. El codo es el índice de la diferencia máxima
    indice_codo = np.argmax(diferencias)
    return k_valores[indice_codo]

def obtener_modelos_e_inercias(X, k_lista, ModelClass, init_func, semilla=42):
    modelos_entrenados = {}
    inercias = []
    
    for k in k_lista:
        print(f"  Entrenando k={k}...")
        modelo = ModelClass(p_cluster_number=k, p_init=init_func, p_tol=1e-4)
        
        modelo.fit(X, p_seed=semilla) 
        
        inercias.append(modelo.inertia_)
        modelos_entrenados[k] = modelo
        
    return modelos_entrenados, inercias

def entrenar_modelo_optimo_multiresolucion(X: np.ndarray, ModelClass, init_func, pasos: list[int], k_min: int = 2, k_max: int = 200, p_seed: int = 42):
    """
    Búsqueda dinámica de K: itera sobre una lista de 'pasos' para ir acotando
    la resolución alrededor del codo detectado.
    """
    # Almacén global (caché) para no repetir entrenamientos jamás
    todos_los_modelos = {}
    todas_las_inercias = {}
    
    current_min = k_min
    current_max = k_max
    codo_actual = k_min
    
    for i, paso in enumerate(pasos):
        print(f"\n--- Iteración {i+1}: Paso de {paso} | Zoom en rango: [{current_min}, {current_max}] ---")
        
        # 1. Generar los K a evaluar en esta resolución
        k_valores = list(range(current_min, current_max + 1, paso))
        
        # Aseguramos que el extremo superior se evalúe siempre para anclar la curva
        if k_valores[-1] != current_max:
            k_valores.append(current_max)
            
        # 2. Filtrar los K que ya hemos entrenado en iteraciones anteriores
        k_nuevos = [k for k in k_valores if k not in todos_los_modelos]
        
        # 3. Entrenar solo los modelos que falten
        if k_nuevos:
            modelos_nuevos, inercias_nuevas = obtener_modelos_e_inercias(X, k_nuevos, ModelClass, init_func, p_seed)
            todos_los_modelos.update(modelos_nuevos)
            todas_las_inercias.update(zip(k_nuevos, inercias_nuevas))
        else:
            print("  (Todos los modelos de esta resolución ya estaban cacheados)")
        
        # 4. Reconstruir la curva GLOBAL con todos los puntos calculados hasta ahora
        k_evaluados = list(todas_las_inercias.keys())
        k_evaluados.sort() # Importantísimo: Kneedle necesita la curva en orden secuencial
        inercias_evaluadas = [todas_las_inercias[k] for k in k_evaluados]
        
        # 5. Aplicar Kneedle sobre la curva cada vez más definida
        codo_actual = kneedle_codo(k_evaluados, inercias_evaluadas)
        print(f"-> Codo detectado con resolución actual: K = {codo_actual}")
        
        # 6. Recalcular límites para el SIGUIENTE paso (Acotamos alrededor del codo actual)
        # Nos abrimos "un paso de la iteración actual" por arriba y por abajo
        current_min = max(k_min, codo_actual - paso)
        current_max = min(k_max, codo_actual + paso)
        
    print(f"\n=======================================================")
    print(f"¡Codo definitivo encontrado: K = {codo_actual}!")
    print(f"Inercia final: {todas_las_inercias[codo_actual]:.4f}")
    print(f"Modelos totales entrenados: {len(todos_los_modelos)} (En lugar de {k_max - k_min + 1})")
    print(f"=======================================================\n")

    # (Justo después de que imprima: "¡Codo definitivo encontrado: K = ...!")
    import matplotlib.pyplot as plt
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_evaluados, inercias_evaluadas, marker='o', linestyle='-', color='b', label='Inercia')
    plt.axvline(x=codo_actual, color='r', linestyle='--', label=f'Codo Detectado (K={codo_actual})')
    plt.title('Detección Analítica del Codo (Kneedle)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia')
    plt.legend()
    plt.grid(True)
    plt.savefig('modelos_finales_todos/grafica_codo.png')
    plt.close()
    
    return todos_los_modelos[codo_actual]

def main(p_instance_path, p_vector_path, p_model, p_init, p_seed = 42, p_varianza_pca = None, p_n_textos = 0,
         p_k_max = 100, p_pasos = (10, 3, 1)):
    # 1. CARGAMOS LOS DATOS (Al estilo de tu compañera)
    print("Cargando dataset...")
    df = pd.read_csv(p_instance_path)
    
    ruta_matriz_ganadora = p_vector_path
    
    if ruta_matriz_ganadora.endswith(".npz"):
        X_global = sparse.load_npz(ruta_matriz_ganadora).tocsr()
    else:
        X_global = np.load(ruta_matriz_ganadora)

    os.makedirs("modelos_finales_todos", exist_ok=True)
    
    # 2. TODOS LOS TEXTOS JUNTOS: un único modelo con una única K
    # Fuera las instancias con vector vacío (igual que en la comparativa)
    df = df[(X_global.getnnz(axis=1) > 0) if sparse.issparse(X_global) else np.any(X_global, axis=1)]
    if 0 < p_n_textos < len(df):
        df = df.loc[np.sort(np.random.default_rng(p_seed).choice(df.index, p_n_textos, replace=False))]
    print(f"Total de instancias: {len(df)} ({df['abreviatura'].nunique()} abreviaturas, "
          f"{df['sentido'].nunique()} sentidos)")

    X = X_global[df.index.to_numpy()]
    if sparse.issparse(X):
        X = X[:, X.getnnz(axis=0) > 0].toarray()
    X = np.asarray(X, dtype=np.float64)
    # La ganadora de la comparativa lleva PCA (p. ej. bert_media+pca80%): no hay archivo con el PCA hecho,
    # se aplica aquí con el mismo PCA de la comparativa
    if p_varianza_pca:
        X = pca(X, p_varianza_pca)
        print(f"PCA al {p_varianza_pca:.0%} de la varianza: {X.shape[1]} dimensiones")

    # 3. BÚSQUEDA DE K CON EL CODO (Kneedle multirresolución)
    modelo_final = entrenar_modelo_optimo_multiresolucion(
        X=X,
        ModelClass=p_model,
        init_func=p_init,
        pasos=list(p_pasos),
        k_min=2,
        k_max=min(p_k_max, len(X) - 1),
        p_seed=p_seed
    )

    # 4. GUARDAR Y EVALUAR EL MODELO FINAL
    ruta_guardado = f"modelos_finales_todos/todos_k{modelo_final.cluster_number}.json"
    modelo_final.save_model(ruta_guardado)
    etiquetas = modelo_final.predict(X)
    df.assign(cluster=etiquetas)[["abreviatura", "sentido", "cluster"]].to_csv("modelos_finales_todos/clusters.csv")

    # Métricas internas (no usan etiquetas) y, solo como referencia, NMI frente a sentidos y abreviaturas
    resultados = evaluar_clustering(modelo_final.metric.format_data(X), etiquetas)
    resultados["ARI_por_abreviatura"] = ari_por_abreviatura(df["abreviatura"].to_numpy(), df["sentido"].to_numpy(),
                                                            etiquetas)
    print(f"ARI dentro de cada abreviatura (media): {resultados['ARI_por_abreviatura']:.4f}")
    print(f"NMI con los sentidos:     {nmi(df['sentido'].to_numpy(), etiquetas):.4f}")
    print(f"NMI con las abreviaturas: {nmi(df['abreviatura'].to_numpy(), etiquetas):.4f}")
    print(f"Modelo guardado en {ruta_guardado}; cluster de cada instancia en modelos_finales_todos/clusters.csv")
    return resultados

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del K-Means final con todos los textos y una única K")
    parser.add_argument("--datos", type=str, required=True, help="Ruta a la matriz vectorizada ganadora (ej. representaciones/X_bert_media.npy)")
    parser.add_argument("--varianza-pca", type=float, default=None, help="PCA a aplicar (ej. 0.8 para bert_media+pca80%%); sin él, no se aplica")
    parser.add_argument("--n-textos", type=int, default=0, help="Textos al azar con los que entrenar (0 = todos)")
    parser.add_argument("--k-max", type=int, default=100, help="Mayor K que se busca")
    parser.add_argument("--pasos", type=int, nargs="+", default=[10, 3, 1], help="Pasos de la búsqueda multirresolución")
    parser.add_argument("--instancias", type=str, default="representaciones/instancias.csv", help="Ruta al CSV con la metadata")
    parser.add_argument("--distancia", choices=["cosine", "euclidean"], default="cosine", help="Métrica geométrica a optimizar")
    parser.add_argument("--init", choices=["random", "plus", "parallel"], default="random", help="Algoritmo de inicialización de centroides")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--oversampling", type=int, default=-1)
    parser.add_argument("--rounds", type=int, default=-1)
    args = parser.parse_args()

    ModelClass = SphericalKMeans if args.distancia == "cosine" else EuclideanKMeans
    
    # KMeans_parallel solo se crea si se elige (con los -1 por defecto lanza un error); -1 = sus valores por defecto
    dict_init = {
        "plus": lambda: kmeans_plus_plus,
        "random": lambda: random_cluster_selection,
        "parallel": lambda: KMeans_parallel(p_oversampling_factor=None if args.oversampling == -1 else args.oversampling,
                                            p_rounds=None if args.rounds == -1 else args.rounds).initialize
    }

    init_func_elegida = dict_init[args.init]()

    main(args.instancias, args.datos, ModelClass, init_func_elegida, args.semilla, args.varianza_pca, args.n_textos,
         args.k_max, args.pasos)
