import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# Configuration
DELTA = 27
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

def process_image(image_path, lower_bound, upper_bound):
    img = Image.open(image_path)
    img_array = np.array(img)
    avg_color_matrix = np.mean(img_array, axis=2) if img_array.ndim == 3 else img_array
    binary_matrix = calculate_binary_mask(avg_color_matrix, lower_bound, upper_bound)
    mean_binary_vector = calculate_mean_binary_vector(binary_matrix)
    return binary_matrix, mean_binary_vector

# Main Workflow
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

# Calculate mean column values for Class 1
if "class1" not in results:
    raise KeyError("'class1' not found in results. Check image availability.")
class1_color_matrices = [img["color_matrix"] for img in results["class1"]]
sum_columns = np.sum([np.sum(matrix, axis=0) for matrix in class1_color_matrices], axis=0)
mean_column_values = sum_columns / (len(class1_color_matrices) * class1_color_matrices[0].shape[0])
print("\nMean Column Values for Class 1:", mean_column_values)

# Define boundaries
lower_bound = mean_column_values - DELTA
upper_bound = mean_column_values + DELTA

# Generate binary masks
binary_masks = {}
for class_name, images in results.items():
    binary_masks[class_name] = []
    for img in images:
        color_matrix = img["color_matrix"]
        binary_mask = calculate_binary_mask(color_matrix, lower_bound, upper_bound)
        binary_masks[class_name].append({
            "file_name": img["file_name"],
            "binary_mask": binary_mask
        })

# Visualize binary masks
for class_name, masks in binary_masks.items():
    for mask_info in masks[:1]:  # Show the first mask for each class
        plt.imshow(mask_info["binary_mask"], cmap="gray")
        plt.title(f"Binary Mask for {class_name} - {mask_info['file_name']}")
        plt.show()

# Calculate mean binary vectors for each class
mean_binary_vectors = {}
for class_name, masks in binary_masks.items():
    combined_sums = np.sum([np.sum(mask["binary_mask"], axis=0) for mask in masks], axis=0)
    total_rows = sum(mask["binary_mask"].shape[0] for mask in masks)
    mean_binary_vector = (combined_sums > (total_rows / 2)).astype(int)
    mean_binary_vectors[class_name] = mean_binary_vector
    print(f"\nMean Binary Vector for {class_name}: {mean_binary_vector}")

# Calculate Hamming distances
hamming_distances = {}
for class_a, vector_a in mean_binary_vectors.items():
    hamming_distances[class_a] = {}
    for class_b, vector_b in mean_binary_vectors.items():
        if class_a != class_b:
            hamming_distances[class_a][class_b] = np.sum(vector_a != vector_b)

# Find closest pairs
closest_pairs = {class_a: min(dists.items(), key=lambda x: x[1]) for class_a, dists in hamming_distances.items()}
print("\nClosest Pairs:")
for class_a, (closest_class, distance) in closest_pairs.items():
    print(f"{class_a} -> Closest: {closest_class}, Distance: {distance}")

# KFE Calculation for all radii
kfe_results = []
best_radius_per_class = {}
max_radius = len(mean_binary_vectors["class1"])
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
        d1_b = t_D1 - t_Betta
        kfe = d1_b * math.log((1 + d1_b + 0.1) / (1 - d1_b + 0.1)) / math.log(2) if abs(d1_b) > 1e-10 else 0
        kfe_results.append({"radius": radius, "our_class": class_a, "D1": t_D1, "betta": t_Betta, "kfe": kfe})
        # Update best radius for each class
        if t_D1 >= 0.5 and t_Betta < 0.5:
            if class_a not in best_radius_per_class or kfe > best_radius_per_class[class_a]["kfe"]:
                best_radius_per_class[class_a] = {"radius": radius, "kfe": kfe}

# Ensure all classes have valid entries in best_radius_per_class
for class_name in results.keys():
    if class_name not in best_radius_per_class:
        best_radius_per_class[class_name] = {"radius": 0, "kfe": 0}

# Plot KFE with Valid Zone
kfe_results_by_class = {}
for result in kfe_results:
    class_name = result["our_class"]
    if class_name not in kfe_results_by_class:
        kfe_results_by_class[class_name] = []
    kfe_results_by_class[class_name].append(result)

for class_name, kfe_data in kfe_results_by_class.items():
    radii = [d["radius"] for d in kfe_data]
    kfes = [d["kfe"] for d in kfe_data]
    valid_zone = [(d["D1"] >= 0.5 and d["betta"] < 0.5) for d in kfe_data]

    plt.figure(figsize=(10, 6))
    plt.plot(radii, kfes, label="KFE", color="blue")

    # Highlight valid zones
    for i in range(len(valid_zone) - 1):
        if valid_zone[i]:
            plt.axvspan(radii[i], radii[i + 1], color="green", alpha=0.3)

    # Mark the best radius
    best_radius = best_radius_per_class[class_name]["radius"]
    if best_radius > 0:
        best_kfe = best_radius_per_class[class_name]["kfe"]
        plt.scatter([best_radius], [best_kfe], color="red", label="Best Radius")
        plt.annotate(f"Best Radius: {best_radius}\nKFE: {best_kfe:.5f}",
                     xy=(best_radius, best_kfe),
                     xytext=(best_radius + 1, best_kfe - 0.1),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     fontsize=10,
                     color="red")

    plt.title(f"KFE Dependence on Radius for {class_name}")
    plt.xlabel("Radius")
    plt.ylabel("KFE")
    plt.legend()
    plt.grid()
    plt.show()

# Test Recognition
test_image_path = "/content/4.png"
test_binary_matrix, test_mean_vector = process_image(test_image_path, lower_bound, upper_bound)
print("\nTest Mean Binary Vector:", test_mean_vector)

classification_results = []
for class_name, vector in mean_binary_vectors.items():
    radius = best_radius_per_class[class_name]["radius"]
    if radius > 0:  # Only consider classes with valid radii
        distance = np.sum(test_mean_vector != vector)
        F_dist = 1 - (distance / radius)
        classification_results.append((class_name, F_dist))

# Sort and display classification results
classification_results.sort(key=lambda x: x[1], reverse=True)
print("\nClassification Results:")
for class_name, F_dist in classification_results:
    print(f"Class: {class_name}, F_dist: {F_dist:.5f}")

# Identify the best match
if classification_results:
    best_class, best_F_dist = classification_results[0]
    if best_F_dist < 0:
        print("\nThe image is classified as UNKNOWN.")
    else:
        print(f"\nThe image is classified as: {best_class} with F_dist = {best_F_dist:.5f}")
else:
    print("\nNo valid classes for classification.")
