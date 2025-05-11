import os
from PIL import Image
import lpips
from torchvision.transforms.functional import to_tensor, resize
from itertools import combinations
import argparse
from tqdm import tqdm

# Step 0. Define the LPIPS function
lpips_fn = lpips.LPIPS(net='vgg').cuda()


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg','tif')):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                img = img.convert('RGB')  # Ensure image is in RGB format
                img = resize(to_tensor(img), (224, 224)).unsqueeze(0).cuda()  # Resize and convert to tensor, then move to GPU
                images.append(img)
    return images


def intra_lpips(X, cluster_centers):
    # Step 1. Assign images to the closest center
    clusters = [[] for _ in range(len(cluster_centers))]
    for image in tqdm(X, desc="Assigning images to clusters"):
        distances = [lpips_fn(image, center).item() for center in cluster_centers]
        closest_cluster_index = distances.index(min(distances))
        clusters[closest_cluster_index].append(image)

    # Step 2. Compute Intra-LPIPS
    all_lpips_distances = []
    for i, cluster in enumerate(clusters):
        for img_i, img_j in tqdm(combinations(cluster, 2), desc=f"Computing Intra-LPIPS for cluster {i + 1}"):
            lpips_distance = lpips_fn(img_i, img_j).item()
            all_lpips_distances.append(lpips_distance)

    return sum(all_lpips_distances) / len(all_lpips_distances) if all_lpips_distances else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Average Intra-LPIPS over Clusters")
    parser.add_argument("--generated_images_folder", type=str, required=True,help="Path to folder containing generated images")
    parser.add_argument("--cluster_centers_folder", type=str, required=True,help="Path to folder containing cluster center images")
    args = parser.parse_args()

    generated_images_folder = args.generated_images_folder
    cluster_centers_folder = args.cluster_centers_folder

    X = load_images_from_folder(generated_images_folder)
    cluster_centers = load_images_from_folder(cluster_centers_folder)

    avg_intra_lpips = intra_lpips(X, cluster_centers)
    print(f"Average Intra-LPIPS over 10 clusters: {avg_intra_lpips}")