import numpy as np
from skimage.color import rgb2lab
import os

# -----------------------------
# Configuration
# -----------------------------

NUM_COLORS = 24
RGB_LEVELS = [64, 128, 192]

BRIGHTNESS_FACTORS = [0.7, 1.0, 1.3]
TEMP_SHIFTS = [-20, 0, 20]
CONTRAST_FACTORS = [0.8, 1.0, 1.2]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)

# -----------------------------
# Generate Base Colors
# -----------------------------

def generate_base_colors(num_colors):
    grid = np.array(
        [[r, g, b] for r in RGB_LEVELS for g in RGB_LEVELS for b in RGB_LEVELS]
    )
    indices = np.random.choice(len(grid), num_colors, replace=False)
    return grid[indices].astype(np.float32)

# -----------------------------
# Illumination Perturbations
# -----------------------------

def brightness_scale(rgb, factor):
    return np.clip(rgb * factor, 0, 255)

def temperature_shift(rgb, shift):
    r, g, b = rgb
    return np.clip([r + shift, g, b - shift], 0, 255)

def contrast_scale(rgb, factor):
    return np.clip((rgb - 128) * factor + 128, 0, 255)

# -----------------------------
# Apply Perturbations
# -----------------------------

def apply_perturbations(colors):
    perturbed_colors = []
    labels = []

    for i, rgb in enumerate(colors):
        for f in BRIGHTNESS_FACTORS:
            perturbed_colors.append(brightness_scale(rgb, f))
            labels.append(("brightness", f))

        for s in TEMP_SHIFTS:
            perturbed_colors.append(temperature_shift(rgb, s))
            labels.append(("temperature", s))

        for c in CONTRAST_FACTORS:
            perturbed_colors.append(contrast_scale(rgb, c))
            labels.append(("contrast", c))

    return np.array(perturbed_colors), labels

# -----------------------------
# RGB to LAB Conversion
# -----------------------------

def rgb_to_lab(colors_rgb):
    rgb_norm = colors_rgb / 255.0
    return rgb2lab(rgb_norm)

# -----------------------------
# Main Execution
# -----------------------------

def main():
    base_colors = generate_base_colors(NUM_COLORS)
    perturbed_colors, perturbation_labels = apply_perturbations(base_colors)

    base_colors_lab = rgb_to_lab(base_colors)
    perturbed_colors_lab = rgb_to_lab(perturbed_colors)

    np.save(os.path.join(DATA_DIR, "colors_original_rgb.npy"), base_colors)
    np.save(os.path.join(DATA_DIR, "colors_original_lab.npy"), base_colors_lab)

    np.save(os.path.join(DATA_DIR, "colors_perturbed_rgb.npy"), perturbed_colors)
    np.save(os.path.join(DATA_DIR, "colors_perturbed_lab.npy"), perturbed_colors_lab)

    np.save(
        os.path.join(DATA_DIR, "perturbation_labels.npy"),
        np.array(perturbation_labels, dtype=object),
    )

    print("Day 2 data generation complete.")
    print(f"Base colors shape (RGB): {base_colors.shape}")
    print(f"Perturbed colors shape (RGB): {perturbed_colors.shape}")

if __name__ == "__main__":
    main()
