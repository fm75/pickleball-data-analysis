import numpy as np
from sklearn.cluster import KMeans

def cluster_holes(circles: np.ndarray, n_clusters: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster detected holes by column (or row) using K-means.
    
    Args:
        circles: Array of shape (N, 3) with (x, y, radius) from detect_holes
        n_clusters: Number of columns/rows to cluster into
    
    Returns:
        tuple of (labels, cluster_centers) where labels is array of cluster 
        assignments for each circle, cluster_centers is (n_clusters, 2) array
    
    Note:
        For vertical pegboard columns, cluster on x-coordinates.
        For horizontal rows, cluster on y-coordinates.
        n_clusters doesn't need to match actual columns exactly - even with
        fewer clusters, linear fitting to each cluster can work well.
    """
    # Extract x coordinates for clustering (use y for horizontal rows)
    X = circles[:, 0].reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return labels