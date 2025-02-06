import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import math

# Configuration
DELTA_MAX = 100  # Maximum delta value
MAX_IMAGES = 1
CLASS_PATHS = {
    "class1": "/content/sample_data/1",
    "class2": "/content/sample_data/2",
    "class3": "/content/sample_data/3"
}

# Helper Functions
def calculate_binary_mask(color_matrix, lower_bound, upper_bound):
    return ((color_matrix >= lower_bound) & (color_matrix <= upper_bound)).astype(int)

def calculate_mean_binary_vector(binary_matrix):
    column_sums = np.sum(binary_matrix, axis=0)
    return (column_sums > (binary_matrix.shape[0] / 2)).astype(int)

def calculate_kfe(t_D1, t_Betta):
    d1_b = t_D1 - t_Betta
    if abs(d1_b) < 1e-10:  # Avoid log(0) errors
        return 0
    return d1_b * math.log((1 + d1_b + 0.1) / (1 - d1_b + 0.1)) / math.log(2)

# Load images and compute color matrices
results = {}
for class_name, path in CLASS_PATHS.items():
    image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < MAX_IMAGES:
        print(f"Skipping {class_name}: Not enough images.")
        continue
    results[class_name] = []
    for img_file in image_files[:MAX_IMAGES]:
        img_path = os.path.join(path, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        avg_color_matrix = np.mean(img_array, axis=2) if img_array.ndim == 3 else img_array
        results[class_name].append({
            "file_name": img_file,
            "color_matrix": avg_color_matrix
        })

# Ensure results are populated
if not results:
    raise ValueError("No valid classes were found with enough images.")

# Store averaged KFE and valid delta points
avg_kfe_values = []
valid_delta_points = []
delta_logs = []

# Main loop for delta optimization
for delta in range(1, DELTA_MAX + 1):
    #print(f"\nProcessing Delta: {delta}")

    # Calculate boundaries for each class
    mean_column_values = {
        class_name: np.mean([img["color_matrix"] for img in images], axis=(0, 1))
        for class_name, images in results.items()
    }
    lower_bounds = {class_name: mean - delta for class_name, mean in mean_column_values.items()}
    upper_bounds = {class_name: mean + delta for class_name, mean in mean_column_values.items()}

    # Generate binary masks and mean binary vectors
    binary_masks = {}
    mean_binary_vectors = {}
    for class_name, images in results.items():
        binary_masks[class_name] = []
        for img in images:
            color_matrix = img["color_matrix"]
            binary_mask = calculate_binary_mask(color_matrix, lower_bounds[class_name], upper_bounds[class_name])
            binary_masks[class_name].append({
                "file_name": img["file_name"],
                "binary_mask": binary_mask
            })
        combined_sums = np.sum([np.sum(mask["binary_mask"], axis=0) for mask in binary_masks[class_name]], axis=0)
        total_rows = sum(mask["binary_mask"].shape[0] for mask in binary_masks[class_name])
        mean_binary_vector = (combined_sums > (total_rows / 2)).astype(int)
        mean_binary_vectors[class_name] = mean_binary_vector

    # Calculate Hamming distances and closest pairs
    hamming_distances = {}
    closest_pairs = {}
    for class_a, vector_a in mean_binary_vectors.items():
        hamming_distances[class_a] = {}
        for class_b, vector_b in mean_binary_vectors.items():
            if class_a != class_b:
                hamming_distances[class_a][class_b] = np.sum(vector_a != vector_b)
        closest_class = min(hamming_distances[class_a], key=hamming_distances[class_a].get)
        closest_pairs[class_a] = (closest_class, hamming_distances[class_a][closest_class])

    # KFE Calculation for all radii
    kfe_results = []
    best_radius_per_class = {}
    max_radius = len(next(iter(mean_binary_vectors.values())))
    for radius in range(max_radius):
        for class_a, (class_b, _) in closest_pairs.items():
            our_matrices = [mask["binary_mask"] for mask in binary_masks[class_a]]
            other_matrices = [mask["binary_mask"] for mask in binary_masks[class_b]]
            our_mean_vector = mean_binary_vectors[class_a]
            k1 = np.sum([np.sum(np.sum(matrix != our_mean_vector, axis=1) <= radius) for matrix in our_matrices])
            k2 = np.sum([np.sum(np.sum(matrix != our_mean_vector, axis=1) <= radius) for matrix in other_matrices])
            total_rows_our = sum(matrix.shape[0] for matrix in our_matrices)
            total_rows_other = sum(matrix.shape[0] for matrix in other_matrices)
            t_D1 = k1 / total_rows_our if total_rows_our > 0 else 0
            t_Betta = k2 / total_rows_other if total_rows_other > 0 else 0
            kfe = calculate_kfe(t_D1, t_Betta)
            kfe_results.append({"radius": radius, "our_class": class_a, "D1": t_D1, "betta": t_Betta, "kfe": kfe})

            # Track maximum KFE for valid conditions
            if t_D1 >= 0.5 and t_Betta < 0.5:
                if class_a not in best_radius_per_class or kfe > best_radius_per_class[class_a]["kfe"]:
                    best_radius_per_class[class_a] = {"radius": radius, "kfe": kfe}

    # Log results for the current delta
    for class_name in results.keys():
        if class_name in best_radius_per_class:
            best_result = best_radius_per_class[class_name]
            delta_logs.append({
                "Delta": delta,
                "Class": class_name,
                "Best Radius": best_result["radius"],
                "Highest KFE": best_result["kfe"]
            })
        else:
            max_kfe = max([entry["kfe"] for entry in kfe_results if entry["our_class"] == class_name], default=0)
            delta_logs.append({
                "Delta": delta,
                "Class": class_name,
                "Best Radius": 0,
                "Highest KFE": max_kfe
            })

    # Check if all classes have valid radii
    all_classes_valid = all(class_name in best_radius_per_class for class_name in results.keys())

    # Store average KFE and mark valid delta
    avg_kfe = np.mean([entry["kfe"] for entry in best_radius_per_class.values()] if best_radius_per_class else [0])
    avg_kfe_values.append(avg_kfe)
    if all_classes_valid:
        valid_delta_points.append((delta, avg_kfe))

# Convert logs to a DataFrame for display
delta_logs_df = pd.DataFrame(delta_logs)

# Display logs
print("\nDelta Optimization Logs:")
print(delta_logs_df)

# Plot KFE vs Delta
plt.figure(figsize=(12, 8))

# Plot the average KFE line
plt.plot(range(1, len(avg_kfe_values) + 1), avg_kfe_values, label="Average KFE", color="blue", linewidth=2)

# Highlight valid area as a shaded region
if valid_delta_points:
    valid_deltas, valid_kfes = zip(*valid_delta_points)
    plt.fill_between(
        range(1, DELTA_MAX + 1),
        0,
        avg_kfe_values,
        where=[delta in valid_deltas for delta in range(1, DELTA_MAX + 1)],
        color="green",
        alpha=0.3,
        label="Valid Area (All Classes)"
    )

# Highlight the best delta with a single point
if valid_delta_points:
    best_delta, best_kfe = max(valid_delta_points, key=lambda x: x[1])
    plt.scatter([best_delta], [best_kfe], color="red", label="Best Delta", zorder=5)
    #plt.annotate(
    #    f"Best Delta: {best_delta}\nKFE: {best_kfe:.5f}",
    #    xy=(best_delta, best_kfe),
    #    xytext=(best_delta + 10, best_kfe - 0.1),
    #    arrowprops=dict(facecolor='black', arrowstyle="->"),
    #    fontsize=12,
    #    color="red"
    #)

# Add title, labels, and legend
plt.title("Average KFE vs Delta", fontsize=16)
plt.xlabel("Delta", fontsize=14)
plt.ylabel("Average KFE", fontsize=14)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid(True)
plt.show()

# Print the best delta
if valid_delta_points:
    print(f"\nBest Delta: {best_delta}, Maximum Average KFE: {best_kfe:.5f}")
else:
    print("\nNo delta value satisfies the conditions for all classes.")
