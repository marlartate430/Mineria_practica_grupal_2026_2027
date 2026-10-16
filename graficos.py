import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_vectors(p_X: np.ndarray, p_n_components: int, p_model):
    # 1. Reducir tus embeddings (ya normalizados) a 2D mediante PCA
    pca = PCA(n_components=p_n_components)

    reduced_X = pca.fit_transform(p_X) 
    
    # Obligatorio: tras el PCA, volvemos a proyectar a la circunferencia de radio 1
    reduced_X = reduced_X / (np.linalg.norm(reduced_X, axis=1, keepdims=True) + 1e-10)
    
    # 2. Configurar el gráfico 2D
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    
    # 3. Dibujar la circunferencia unidad de fondo
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.5, label='Circunferencia Unidad')
    
    # 4. Pintar los puntos por colores (según tus predicciones de predict)
    etiquetas = p_model.predict(p_X)
    
    # Generamos una lista de colores espaciados uniformemente
    colores = plt.cm.hsv(np.linspace(0, 1, p_model.cluster_number + 1))
    
    for cluster_id in range(p_model.cluster_number):
        mascara = (etiquetas == cluster_id)
        # IMPORTANTE: Usamos reduced_X para pintar, no p_X
        ax.scatter(reduced_X[mascara, 0], reduced_X[mascara, 1], 
                   color=colores[cluster_id], label=f'Cluster {cluster_id}', s=25, alpha=0.8, edgecolors='none')
    
    # 5. Transformar y pintar los centroides (proyectados a la circunferencia)
    reduced_centroids = pca.transform(p_model.centroids)
    reduced_centroids = reduced_centroids / (np.linalg.norm(reduced_centroids, axis=1, keepdims=True) + 1e-10)
    
    ax.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
               color='black', marker='*', s=250, edgecolor='white', linewidth=1.5, zorder=5, label='Centroides')
    
    # 6. Ajustar límites, ejes y etiquetas
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal') # Fuerza que el círculo sea redondo y no se achate
    ax.set_xlabel('Componente PCA 1')
    ax.set_ylabel('Componente PCA 2')
    ax.set_title("Visualización de vectores en el plano 2D tras PCA")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Colocar la leyenda fuera para que no tape los puntos
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
