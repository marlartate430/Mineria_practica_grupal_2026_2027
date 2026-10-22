import numpy as np
import pandas as pd
from scipy import sparse
import os
import argparse
from kmeans_init import kmeans_plus_plus, random_cluster_selection, KMeans_parallel

from modeloKMeans import SphericalKMeans, EuclideanKMeans
from kmeans_init import kmeans_plus_plus

parallel_dict = {}

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

def obtener_modelos_e_inercias(X, k_lista, ModelClass, init_func, semilla=42, p_rounds=1):
    modelos_entrenados = {}
    inercias = []

    global parallel_dict

    for k in k_lista:
        print(f"  Entrenando k={k}...")

        if init_func == "parallel":
            if not k in parallel_dict:
                parallel_dict[k] = KMeans_parallel(2 * k, p_rounds).initialize
            
            modelo = ModelClass(p_cluster_number=k, p_init=parallel_dict[k], p_tol=1e-4)
        else:
            modelo = ModelClass(p_cluster_number=k, p_init=init_func, p_tol=1e-4)
        
        modelo.fit(X, p_seed=semilla) 
        
        inercias.append(modelo.inertia_)
        modelos_entrenados[k] = modelo
        
    return modelos_entrenados, inercias

def entrenar_modelo_optimo_multiresolucion(X: np.ndarray, ModelClass, init_func, pasos: list[int], k_min: int = 2, k_max: int = 200, p_seed: int = 42, p_rounds=1):
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
            modelos_nuevos, inercias_nuevas = obtener_modelos_e_inercias(X, k_nuevos, ModelClass, init_func)
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
    plt.savefig('grafica_codo.png')
    plt.close()
    
    return todos_los_modelos[codo_actual]

def main(p_instance_path, p_vector_path, p_model, p_init, p_seed = 42, p_rounds=1):
    # 1. CARGAMOS LOS DATOS (Al estilo de tu compañera)
    print("Cargando dataset...")
    df = pd.read_csv(p_instance_path)
    
    # Asume que ya mirasteis los resultados de ella y ganó, por ejemplo, TF-IDF
    ruta_matriz_ganadora = p_vector_path
    
    if ruta_matriz_ganadora.endswith(".npz"):
        X_global = sparse.load_npz(ruta_matriz_ganadora)
    else:
        X_global = np.load(ruta_matriz_ganadora)

    os.makedirs("modelos_finales", exist_ok=True)
    
    print(f"Total de instancias: {len(df)}")
    
    # 2. BUCLE POR ABREVIATURA (El terreno de tu compañera)
    for abrev, grupo in df.groupby("abreviatura"):
        print(f"\n--- Analizando a ciegas: {abrev} (Textos: {len(grupo)}) ---")
        
        # Filtramos solo los textos de esta abreviatura
        indices = grupo.index.to_numpy()
        X_abrev = X_global[indices]
        
        # Magia para que no pete la RAM (lo que hizo tu compañera)
        if sparse.issparse(X_abrev):
            X_abrev = X_abrev[:, X_abrev.getnnz(axis=0) > 0].toarray()
        X_abrev = np.asarray(X_abrev, dtype=np.float64)

        k_max_logico = min(15, len(grupo) - 1)
        if k_max_logico < 2:
            print(f"  Pocos datos para clusterizar. Saltando...")
            continue
            
        # 3. TU MOTOR EN ACCIÓN (Tu terreno)
        try:
            modelo_final = entrenar_modelo_optimo_multiresolucion(
                X=X_abrev,
                ModelClass=p_model, # O EuclideanKMeans, según lo que ganara
                init_func=p_init,
                pasos=[3, 1], 
                k_min=2,
                k_max=k_max_logico,
                p_seed=p_seed,
                p_rounds=p_rounds
            )
            
            # Guardamos el JSON
            ruta_guardado = f"modelos_finales/{abrev}_k{modelo_final.cluster_number}.json"
            modelo_final.save_model(ruta_guardado)
            
        except Exception as e:
            print(f"  Error procesando {abrev}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento y desambiguación masiva con K-Means")
    parser.add_argument("--datos", type=str, required=True, help="Ruta a la matriz vectorizada ganadora (ej. representaciones/X_bert_abrev+pca90%.npy)")
    parser.add_argument("--instancias", type=str, default="representaciones/instancias.csv", help="Ruta al CSV con la metadata")
    parser.add_argument("--distancia", choices=["cosine", "euclidean"], default="cosine", help="Métrica geométrica a optimizar")
    parser.add_argument("--init", choices=["random", "plus", "parallel"], default="plus", help="Algoritmo de inicialización de centroides")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    ModelClass = SphericalKMeans if args.distancia == "cosine" else EuclideanKMeans
    
    dict_init = {
        "plus": kmeans_plus_plus,
        "random": random_cluster_selection,
        "parallel": "parallel",
    }

    init_func_elegida = dict_init[args.init]

    main(args.instancias, args.datos, ModelClass, init_func_elegida, args.semilla, args.rounds)
