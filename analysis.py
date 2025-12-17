import numpy as np
from model import learn_perceptual_embedding

DATA_DIR = "data"

def load_data():
    original_lab = np.load(f"{DATA_DIR}/colors_original_lab.npy")
    perturbed_lab = np.load(f"{DATA_DIR}/colors_perturbed_lab.npy")
    return original_lab, perturbed_lab

def compute_mean_distance(original, perturbed):
    """
    Mean L2 distance between original colors
    and their illumination-perturbed versions.
    """
    num_colors = original.shape[0]
    num_perturbations = perturbed.shape[0] // num_colors

    distances = []

    for i in range(num_colors):
        base = original[i]
        for j in range(num_perturbations):
            idx = i * num_perturbations + j
            distances.append(np.linalg.norm(base - perturbed[idx]))

    return np.mean(distances)

def main():
    original_lab, perturbed_lab = load_data()

    # Baseline invariance in LAB space
    lab_distance = compute_mean_distance(original_lab, perturbed_lab)

    # Learned embedding invariance
    pca, Z = learn_perceptual_embedding(original_lab, perturbed_lab)
    Z_orig = Z[:len(original_lab)]
    Z_pert = Z[len(original_lab):]

    learned_distance = compute_mean_distance(Z_orig, Z_pert)

    print("Mean distance under illumination changes:")
    print(f"CIELAB space: {lab_distance:.4f}")
    print(f"Learned embedding: {learned_distance:.4f}")

if __name__ == "__main__":
    main()
