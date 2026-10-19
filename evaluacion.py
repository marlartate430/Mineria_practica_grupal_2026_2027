import numpy as np

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def evaluar_clustering(X, labels, kmeans=None):
    """
    Evalúa un clustering mediante métricas internas de scikit-learn.

    Parámetros
    ----------
    X : array-like, shape (n_samples, n_features)
        Datos utilizados para realizar el clustering.

    labels : array-like, shape (n_samples,)
        Etiquetas de clúster asignadas a cada muestra.

    kmeans : sklearn.cluster.KMeans, opcional
        Modelo KMeans entrenado. Si se proporciona, se calcula
        también la inercia del modelo.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    # Comprobar dimensiones
    if X.ndim != 2:
        raise ValueError("X debe ser un array bidimensional.")

    if labels.ndim != 1:
        raise ValueError("labels debe ser un array unidimensional.")

    if len(X) != len(labels):
        raise ValueError(
            "X y labels deben contener el mismo número de muestras."
        )

    # Número de clústeres
    n_clusters = len(np.unique(labels))
    n_samples = len(X)

    if n_clusters < 2:
        raise ValueError(
            "Se necesitan al menos 2 clústeres para evaluar el clustering."
        )

    if n_clusters >= n_samples:
        raise ValueError(
            "El número de clústeres debe ser menor que el número de muestras."
        )

    # Calcular métricas internas
    silhouette = silhouette_score(X, labels)

    calinski_harabasz = calinski_harabasz_score(X, labels)

    davies_bouldin = davies_bouldin_score(X, labels)

    # Resultados
    resultados = {
        "n_muestras": n_samples,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabasz,
        "davies_bouldin_score": davies_bouldin
        "inertia": kmeans.inertia_
    }

    # Mostrar resultados
    print("\n===== EVALUACIÓN DEL CLUSTERING =====")
    print(f"Número de muestras: {n_samples}")
    print(f"Número de clústeres: {n_clusters}")
    print("-" * 40)
    print(f"Silhouette Score:        {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    print(f"Davies-Bouldin Score:    {davies_bouldin:.4f}")
    print(f"Inertia:                 {kmeans.inertia_:.4f}")


    print("=" * 40)
