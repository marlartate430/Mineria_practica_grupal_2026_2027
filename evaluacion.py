
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def evaluar_clustering(
    X,
    labels,
    kmeans=None,
    tiempo_ejecucion=None,
    max_muestras_silhouette=5000,
    random_state=42
):
    """
    Evalúa un clustering mediante métricas internas.

    Parámetros
    ----------
    X : array-like, shape (n_samples, n_features)
        Datos utilizados para realizar el clustering.

    labels : array-like, shape (n_samples,)
        Clúster asignado a cada muestra.

    kmeans : objeto, opcional
        Modelo K-Means entrenado. Se utiliza para consultar
        el número de clústeres configurado, si está disponible.

    tiempo_ejecucion : float, opcional
        Tiempo de ejecución del algoritmo en segundos.

    max_muestras_silhouette : int o None
        Número máximo de muestras utilizadas para calcular
        Silhouette. None utiliza todas las muestras.

    random_state : int
        Semilla para garantizar reproducibilidad.

    Retorna
    -------
    resultados : dict
        Diccionario con las métricas calculadas.
    """

    # --------------------------------------------------
    # 1. VALIDACIÓN DE DATOS
    # --------------------------------------------------

    X = np.asarray(X)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X debe ser bidimensional.")

    if labels.ndim != 1:
        raise ValueError("labels debe ser unidimensional.")

    if len(X) != len(labels):
        raise ValueError(
            "X y labels deben tener el mismo número de muestras."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("X contiene valores NaN o infinitos.")

    n_samples = X.shape[0]
    clusters, cantidades = np.unique(
        labels, return_counts=True
    )

    n_clusters = len(clusters)

    if not 2 <= n_clusters < n_samples:
        raise ValueError(
            "El número de clústeres debe estar entre "
            "2 y n_samples - 1."
        )

    # --------------------------------------------------
    # 2. SILHOUETTE SCORE
    # --------------------------------------------------

    if (
        max_muestras_silhouette is not None
        and n_samples > max_muestras_silhouette
    ):
        if max_muestras_silhouette < n_clusters + 1:
            raise ValueError(
                "La muestra de Silhouette es demasiado pequeña."
            )

        # Muestreo estratificado por clúster: garantiza al menos una muestra de cada clúster.
        rng = np.random.default_rng(random_state)

        indices = [
            rng.choice(np.flatnonzero(labels == cluster))
            for cluster in clusters
        ]

        restantes = np.setdiff1d(
            np.arange(n_samples),
            np.asarray(indices),
            assume_unique=False
        )

        indices.extend(
            rng.choice(
                restantes,
                size=max_muestras_silhouette - n_clusters,
                replace=False
            ).tolist()
        )

        indices = np.asarray(indices)

        silhouette = silhouette_score(
            X[indices],
            labels[indices]
        )

        muestras_silhouette = len(indices)

    else:
        silhouette = silhouette_score(X, labels)
        muestras_silhouette = n_samples

    # --------------------------------------------------
    # 3. CALINSKI-HARABASZ
    # --------------------------------------------------

    calinski_harabasz = calinski_harabasz_score(
        X, labels
    )

    # --------------------------------------------------
    # 4. DAVIES-BOULDIN
    # --------------------------------------------------

    davies_bouldin = davies_bouldin_score(
        X, labels
    )

    # --------------------------------------------------
    # 5. INERCIA
    # --------------------------------------------------

    inertia = 0.0

    for cluster in clusters:
        puntos = X[labels == cluster]

        centroide = np.mean(puntos, axis=0)

        inertia += np.sum(
            (puntos - centroide) ** 2
        )

    # --------------------------------------------------
    # 6. DISTRIBUCIÓN DE CLÚSTERES
    # --------------------------------------------------

    distribucion = {
        str(cluster): int(cantidad)
        for cluster, cantidad in zip(clusters, cantidades)
    }

    cluster_minimo = int(np.min(cantidades))
    cluster_maximo = int(np.max(cantidades))

    # Detectar clústeres vacíos si se conoce K.
    k_esperado = getattr(kmeans, "n_clusters", None)

    clusters_vacios = (
        max(0, k_esperado - n_clusters)
        if k_esperado is not None
        else None
    )

    # --------------------------------------------------
    # 7. GUARDAR RESULTADOS
    # --------------------------------------------------

    resultados = {
        "n_muestras": n_samples,
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabasz,
        "davies_bouldin_score": davies_bouldin,
        "inertia": inertia,
        "cluster_minimo": cluster_minimo,
        "cluster_maximo": cluster_maximo,
        "distribucion": distribucion,
        "clusters_vacios": clusters_vacios,
        "tiempo_ejecucion": tiempo_ejecucion,
        "muestras_silhouette": muestras_silhouette
    }

    # --------------------------------------------------
    # 8. MOSTRAR RESULTADOS
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("          EVALUACIÓN DEL CLUSTERING")
    print("=" * 50)

    print(f"Número de muestras: {n_samples}")
    print(f"Número de clústeres: {n_clusters}")

    print("\n--- MÉTRICAS INTERNAS ---")

    print(f"Silhouette Score:        {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
    print(f"Davies-Bouldin Score:    {davies_bouldin:.4f}")
    print(f"Inercia:                 {inertia:.4f}")

    print("\n--- DISTRIBUCIÓN DE CLÚSTERES ---")

    for cluster, cantidad in distribucion.items():
        porcentaje = cantidad / n_samples * 100

        print(
            f"Clúster {cluster}: "
            f"{cantidad} muestras ({porcentaje:.2f}%)"
        )

    print(f"\nClúster más pequeño: {cluster_minimo}")
    print(f"Clúster más grande:  {cluster_maximo}")

    if clusters_vacios is not None:
        print(f"Clústeres vacíos: {clusters_vacios}")

    if tiempo_ejecucion is not None:
        print(
            f"\nTiempo de ejecución: "
            f"{tiempo_ejecucion:.4f} segundos"
        )

    print(
        f"Muestras utilizadas para Silhouette: "
        f"{muestras_silhouette}"
    )

    print("=" * 50)

    return resultados
