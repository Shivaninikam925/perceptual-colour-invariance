import numpy as np
from sklearn.decomposition import PCA

def learn_perceptual_embedding(
    original_lab,
    perturbed_lab,
    embedding_dim=2
):
    """
    Learn a minimal linear embedding that captures
    stable directions under illumination changes.
    """

    # Stack original and perturbed samples
    X = np.vstack([original_lab, perturbed_lab])

    # PCA as a minimal, interpretable baseline
    pca = PCA(n_components=embedding_dim)
    Z = pca.fit_transform(X)

    return pca, Z
