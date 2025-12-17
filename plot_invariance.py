import numpy as np
import matplotlib.pyplot as plt
import os
from model import learn_perceptual_embedding

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def compute_mean_distance(original, perturbed):
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
    # Load data
    original_rgb = np.load(f"{DATA_DIR}/colors_original_rgb.npy")
    perturbed_rgb = np.load(f"{DATA_DIR}/colors_perturbed_rgb.npy")

    original_lab = np.load(f"{DATA_DIR}/colors_original_lab.npy")
    perturbed_lab = np.load(f"{DATA_DIR}/colors_perturbed_lab.npy")

    # RGB invariance
    rgb_dist = compute_mean_distance(original_rgb, perturbed_rgb)

    # CIELAB invariance
    lab_dist = compute_mean_distance(original_lab, perturbed_lab)

    # Learned embedding invariance
    pca, Z = learn_perceptual_embedding(original_lab, perturbed_lab)
    Z_orig = Z[:len(original_lab)]
    Z_pert = Z[len(original_lab):]
    learned_dist = compute_mean_distance(Z_orig, Z_pert)

    representations = ["RGB", "CIELAB", "Learned"]
    distances = [rgb_dist, lab_dist, learned_dist]

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(representations, distances)
    plt.ylabel("Mean Distance Under Illumination Changes")
    plt.title("Perceptual Invariance of Color Representations")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perceptual_invariance.png", dpi=300)
    plt.close()

    print("Saved figure to figures/perceptual_invariance.png")

if __name__ == "__main__":
    main()
