# ==========================================================
# CONTROLLED AUTOENCODER BASELINE FOR SIMCLR COMPARISON
#
# Purpose
# -------
# Train a conventional convolutional autoencoder using the
# exact same 7,415 particles and contour-based preprocessing
# used for the SimCLR framework.
#
# Controlled comparison:
#   - Same image folder
#   - Same 7,415 filenames and row order
#   - Same contour normalisation (fill_fraction = 0.82)
#   - Same 224 x 224 input size
#   - Same 512-dimensional representation
#   - Same random seed
#   - Same K-means configuration
#   - Same diagonal-covariance GMM configuration
#   - Clustering directly on learned embeddings
#   - PCA used only for visualisation
#
# Main outputs
# ------------
# autoencoder_embeddings.npy
# autoencoder_embedding_image_paths.csv
# autoencoder_kmeans_K2_cluster_assignments.csv
# autoencoder_gmm_K2_cluster_assignments.csv
# autoencoder_pca_coordinates.csv
# reconstruction_examples.png
# autoencoder_training_loss.png
# ==========================================================

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

IMAGE_DIR = Path(
    "/mnt/hpccs01/home/n11555581/Code/CLASSIFICATIONFOLDER2"
)

# This CSV defines the exact 7,415 particles used by SimCLR.
# It must contain either image_name or image_path.
REFERENCE_CSV = Path(
    "/mnt/hpccs01/home/n11555581/Code/"
    "SIMCLR_CLASSIFICATION_TRIAL/"
    "simclr_particle_cluster_assignments.csv"
)

OUTPUT_DIR = Path(
    "/mnt/hpccs01/home/n11555581/Code/"
    "AUTOENCODER_SIMCLR_CONTROLLED_COMPARISON"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N_IMAGES = 7415

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
LATENT_DIM = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

NUM_WORKERS = 4
RANDOM_SEED = 42

FINAL_K = 2

# Same GMM settings as the final SimCLR framework
GMM_COVARIANCE_TYPE = "diag"
GMM_N_INIT = 50
GMM_MAX_ITER = 500
GMM_REG_COVAR = 1e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Image folder:", IMAGE_DIR, flush=True)
print("Reference CSV:", REFERENCE_CSV, flush=True)
print("Output folder:", OUTPUT_DIR, flush=True)
print("Using device:", DEVICE, flush=True)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)


# ==========================================================
# 2. REPRODUCIBILITY
# ==========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducible behaviour. This can reduce speed slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ==========================================================
# 3. LOAD THE EXACT 7,415 SIMCLR FILENAMES
# ==========================================================

if not REFERENCE_CSV.exists():
    raise FileNotFoundError(
        f"Reference CSV not found: {REFERENCE_CSV}"
    )

reference_df = pd.read_csv(REFERENCE_CSV)

if "image_name" not in reference_df.columns:
    if "image_path" not in reference_df.columns:
        raise ValueError(
            "REFERENCE_CSV must contain image_name or image_path."
        )

    reference_df["image_name"] = reference_df["image_path"].map(
        os.path.basename
    )

# Preserve the first occurrence and the original CSV order.
reference_df = reference_df.drop_duplicates(
    subset="image_name",
    keep="first"
).reset_index(drop=True)

reference_names = reference_df["image_name"].astype(str).tolist()

if len(reference_names) != EXPECTED_N_IMAGES:
    raise ValueError(
        f"Expected exactly {EXPECTED_N_IMAGES} unique filenames, "
        f"but the reference CSV contains {len(reference_names)}.\n"
        "Use the assignment CSV from the original 7,415-particle "
        "SimCLR run."
    )

image_paths = []
missing_files = []

for image_name in reference_names:
    image_path = IMAGE_DIR / image_name

    if image_path.exists():
        image_paths.append(str(image_path))
    else:
        missing_files.append(image_name)

if missing_files:
    missing_df = pd.DataFrame({
        "missing_image_name": missing_files
    })

    missing_df.to_csv(
        OUTPUT_DIR / "missing_reference_images.csv",
        index=False
    )

    raise FileNotFoundError(
        f"{len(missing_files)} of the required 7,415 images "
        "were not found. See missing_reference_images.csv."
    )

if len(image_paths) != EXPECTED_N_IMAGES:
    raise RuntimeError(
        "The final image list does not contain exactly 7,415 particles."
    )

dataset_manifest = pd.DataFrame({
    "embedding_row": np.arange(EXPECTED_N_IMAGES),
    "image_name": reference_names,
    "image_path": image_paths,
})

dataset_manifest.to_csv(
    OUTPUT_DIR / "autoencoder_dataset_manifest_7415.csv",
    index=False
)

print(
    f"Confirmed exact controlled dataset: {len(image_paths)} images",
    flush=True
)


# ==========================================================
# 4. EXACT SIMCLR CONTOUR-BASED SIZE NORMALISATION
# ==========================================================

def normalize_particle_size_pil(
    img,
    output_size=224,
    padding=5,
    fill_fraction=0.82,
):
    """
    Exact preprocessing used in the controlled SimCLR model:
    detect the centred particle, crop it, resize it while
    preserving aspect ratio, and centre it on a black canvas.
    """
    arr = np.array(img)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(arr, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((5, 5), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return img.resize((output_size, output_size))

    img_h, img_w = arr.shape
    centre_x = img_w / 2
    centre_y = img_h / 2

    best_contour = None
    best_score = -np.inf

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        contour_cx = x + width / 2
        contour_cy = y + height / 2

        distance = np.sqrt(
            (contour_cx - centre_x) ** 2
            + (contour_cy - centre_y) ** 2
        )

        score = area - 2.0 * distance

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return img.resize((output_size, output_size))

    x, y, width, height = cv2.boundingRect(best_contour)

    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + width + padding, arr.shape[1])
    y1 = min(y + height + padding, arr.shape[0])

    cropped = arr[y0:y1, x0:x1]

    crop_h, crop_w = cropped.shape

    scale = (
        output_size * fill_fraction
    ) / max(crop_w, crop_h)

    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC
    )

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=interpolation,
    )

    canvas = np.zeros(
        (output_size, output_size),
        dtype=np.uint8,
    )

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return Image.fromarray(canvas)


# ==========================================================
# 5. DATASET
# ==========================================================

class ControlledParticleDataset(Dataset):
    def __init__(self, ordered_paths, transform=None):
        self.image_paths = list(ordered_paths)
        self.transform = transform

        if len(self.image_paths) != EXPECTED_N_IMAGES:
            raise ValueError(
                "ControlledParticleDataset must contain exactly "
                f"{EXPECTED_N_IMAGES} images."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        image = Image.open(path).convert("L")

        image = normalize_particle_size_pil(
            image,
            output_size=IMG_SIZE,
            padding=5,
            fill_fraction=0.82,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, path


base_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = ControlledParticleDataset(
    ordered_paths=image_paths,
    transform=base_transform,
)

# Generator ensures reproducible shuffled batch order.
loader_generator = torch.Generator()
loader_generator.manual_seed(RANDOM_SEED)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=(DEVICE == "cuda"),
    generator=loader_generator,
)

feature_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=(DEVICE == "cuda"),
)


# ==========================================================
# 6. SAVE PREPROCESSING CHECK
# ==========================================================

check_indices = np.linspace(
    0,
    len(image_paths) - 1,
    8,
    dtype=int,
)

fig, axes = plt.subplots(
    2,
    len(check_indices),
    figsize=(14, 4.5),
)

for column, index in enumerate(check_indices):
    original = Image.open(image_paths[index]).convert("L")

    normalised = normalize_particle_size_pil(
        original,
        output_size=IMG_SIZE,
        padding=5,
        fill_fraction=0.82,
    )

    axes[0, column].imshow(original, cmap="gray")
    axes[0, column].axis("off")

    axes[1, column].imshow(normalised, cmap="gray")
    axes[1, column].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Normalised")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "autoencoder_size_normalisation_check.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ==========================================================
# 7. CONVOLUTIONAL AUTOENCODER
# ==========================================================

class ConvAutoencoder(nn.Module):
    """
    Conventional convolutional autoencoder with a
    512-dimensional bottleneck matching the dimensionality
    of the SimCLR encoder representation used for clustering.
    """

    def __init__(self, latent_dim=512):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = 256 * 14 * 14

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.flatten_dim,
                latent_dim,
            ),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(
                latent_dim,
                self.flatten_dim,
            ),
            nn.ReLU(inplace=True),
        )

        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(
                1,
                (256, 14, 14),
            ),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder_cnn(x)
        z = self.encoder_fc(x)
        return z

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decoder_fc(z)
        reconstruction = self.decoder_cnn(reconstruction)
        return reconstruction, z


model = ConvAutoencoder(
    latent_dim=LATENT_DIM
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ==========================================================
# 8. TRAIN AUTOENCODER
# ==========================================================

loss_history = []

for epoch in range(EPOCHS):
    model.train()

    epoch_loss_sum = 0.0
    n_seen = 0

    for batch_index, (images, _) in enumerate(train_loader):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        reconstructed, _ = model(images)

        loss = criterion(
            reconstructed,
            images,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        epoch_loss_sum += (
            loss.item() * batch_size
        )

        n_seen += batch_size

        if batch_index % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.6f}",
                flush=True,
            )

    epoch_loss = epoch_loss_sum / n_seen
    loss_history.append(epoch_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Mean reconstruction loss: {epoch_loss:.6f}",
        flush=True,
    )

torch.save(
    model.state_dict(),
    OUTPUT_DIR / "autoencoder_model_512d.pth",
)

loss_df = pd.DataFrame({
    "epoch": np.arange(1, EPOCHS + 1),
    "reconstruction_loss": loss_history,
})

loss_df.to_csv(
    OUTPUT_DIR / "autoencoder_training_loss.csv",
    index=False,
)

plt.figure(figsize=(6, 4))
plt.plot(
    loss_df["epoch"],
    loss_df["reconstruction_loss"],
)
plt.xlabel("Epoch")
plt.ylabel("Mean reconstruction loss")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "autoencoder_training_loss.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ==========================================================
# 9. RECONSTRUCTION CHECK
# ==========================================================

model.eval()

images, paths = next(iter(feature_loader))

with torch.no_grad():
    reconstructed, _ = model(
        images.to(DEVICE)
    )

images_np = images.cpu().numpy()
reconstructed_np = reconstructed.cpu().numpy()

n_show = min(8, images_np.shape[0])

fig, axes = plt.subplots(
    2,
    n_show,
    figsize=(14, 4.5),
)

for index in range(n_show):
    axes[0, index].imshow(
        images_np[index, 0],
        cmap="gray",
    )

    axes[1, index].imshow(
        reconstructed_np[index, 0],
        cmap="gray",
    )

    axes[0, index].axis("off")
    axes[1, index].axis("off")

axes[0, 0].set_ylabel("Input")
axes[1, 0].set_ylabel("Reconstructed")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "reconstruction_examples.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ==========================================================
# 10. EXTRACT 512-DIMENSIONAL EMBEDDINGS
# ==========================================================

model.eval()

embedding_batches = []
ordered_output_paths = []

with torch.no_grad():
    for batch_index, (images, paths) in enumerate(feature_loader):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        embeddings = model.encode(images)

        embedding_batches.append(
            embeddings.cpu().numpy()
        )

        ordered_output_paths.extend(paths)

        if batch_index % 10 == 0:
            print(
                f"Extracting embeddings: "
                f"batch {batch_index + 1}/{len(feature_loader)}",
                flush=True,
            )

autoencoder_embeddings = np.vstack(
    embedding_batches
)

if autoencoder_embeddings.shape != (
    EXPECTED_N_IMAGES,
    LATENT_DIM,
):
    raise RuntimeError(
        "Unexpected autoencoder embedding shape: "
        f"{autoencoder_embeddings.shape}"
    )

output_names = [
    os.path.basename(path)
    for path in ordered_output_paths
]

if output_names != reference_names:
    raise RuntimeError(
        "Embedding output order does not match the controlled "
        "7,415-image reference order."
    )

np.save(
    OUTPUT_DIR / "autoencoder_embeddings.npy",
    autoencoder_embeddings,
)

embedding_order_df = pd.DataFrame({
    "embedding_row": np.arange(EXPECTED_N_IMAGES),
    "image_name": output_names,
    "image_path": ordered_output_paths,
})

embedding_order_df.to_csv(
    OUTPUT_DIR / "autoencoder_embedding_image_paths.csv",
    index=False,
)

print(
    "Autoencoder embedding matrix:",
    autoencoder_embeddings.shape,
    flush=True,
)


# ==========================================================
# 11. K-MEANS — DIRECTLY ON AUTOENCODER EMBEDDINGS
# ==========================================================

kmeans = KMeans(
    n_clusters=FINAL_K,
    init="k-means++",
    n_init=50,
    max_iter=500,
    random_state=RANDOM_SEED,
)

kmeans_labels = kmeans.fit_predict(
    autoencoder_embeddings
)

kmeans_distances = kmeans.transform(
    autoencoder_embeddings
)

assigned_kmeans_distance = kmeans_distances[
    np.arange(EXPECTED_N_IMAGES),
    kmeans_labels,
]

kmeans_df = embedding_order_df.copy()

kmeans_df["kmeans_cluster"] = (
    kmeans_labels
)

kmeans_df[
    "distance_to_assigned_centroid"
] = assigned_kmeans_distance

for cluster_index in range(FINAL_K):
    kmeans_df[
        f"distance_to_cluster_{cluster_index}_centroid"
    ] = kmeans_distances[:, cluster_index]

kmeans_df.to_csv(
    OUTPUT_DIR / "autoencoder_kmeans_K2_cluster_assignments.csv",
    index=False,
)


# ==========================================================
# 12. GMM — DIRECTLY ON AUTOENCODER EMBEDDINGS
# ==========================================================

gmm = GaussianMixture(
    n_components=FINAL_K,
    covariance_type=GMM_COVARIANCE_TYPE,
    random_state=RANDOM_SEED,
    n_init=GMM_N_INIT,
    max_iter=GMM_MAX_ITER,
    reg_covar=GMM_REG_COVAR,
)

gmm_labels = gmm.fit_predict(
    autoencoder_embeddings
)

gmm_probabilities = gmm.predict_proba(
    autoencoder_embeddings
)

gmm_confidence = gmm_probabilities.max(
    axis=1
)

gmm_df = embedding_order_df.copy()

gmm_df["gmm_cluster"] = gmm_labels
gmm_df["gmm_confidence"] = gmm_confidence

for cluster_index in range(FINAL_K):
    gmm_df[
        f"posterior_cluster_{cluster_index}"
    ] = gmm_probabilities[:, cluster_index]

gmm_df.to_csv(
    OUTPUT_DIR / "autoencoder_gmm_K2_cluster_assignments.csv",
    index=False,
)


# ==========================================================
# 13. CLUSTERING METRICS FOR THE FOUR-METHOD COMPARISON
# ==========================================================

comparison_metrics = []

for method_name, labels in [
    ("Autoencoder + K-means", kmeans_labels),
    ("Autoencoder + GMM", gmm_labels),
]:
    comparison_metrics.append({
        "method": method_name,
        "n_clusters": FINAL_K,
        "silhouette_score": silhouette_score(
            autoencoder_embeddings,
            labels,
        ),
        "davies_bouldin_score": davies_bouldin_score(
            autoencoder_embeddings,
            labels,
        ),
        "calinski_harabasz_score": calinski_harabasz_score(
            autoencoder_embeddings,
            labels,
        ),
    })

comparison_metrics_df = pd.DataFrame(
    comparison_metrics
)

comparison_metrics_df.to_csv(
    OUTPUT_DIR / "autoencoder_clustering_metrics_K2.csv",
    index=False,
)

print(
    "\nAutoencoder clustering metrics:",
    flush=True,
)

print(
    comparison_metrics_df.to_string(index=False),
    flush=True,
)


# ==========================================================
# 14. PCA FOR VISUALISATION ONLY
# ==========================================================

pca = PCA(
    n_components=2,
    random_state=RANDOM_SEED,
)

embedding_pca = pca.fit_transform(
    autoencoder_embeddings
)

pc1_variance = (
    pca.explained_variance_ratio_[0]
    * 100
)

pc2_variance = (
    pca.explained_variance_ratio_[1]
    * 100
)

pca_df = embedding_order_df.copy()

pca_df["PC1"] = embedding_pca[:, 0]
pca_df["PC2"] = embedding_pca[:, 1]
pca_df["kmeans_cluster"] = kmeans_labels
pca_df["gmm_cluster"] = gmm_labels
pca_df["gmm_confidence"] = gmm_confidence

pca_df.to_csv(
    OUTPUT_DIR / "autoencoder_pca_coordinates.csv",
    index=False,
)


# ==========================================================
# 15. PCA PLOT — K-MEANS AND GMM
# ==========================================================

cluster_colours = {
    0: "#4C72B0",
    1: "#DD8452",
}

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.5),
    sharex=True,
    sharey=True,
)

for axis, labels, title, panel in [
    (
        axes[0],
        kmeans_labels,
        "Autoencoder + K-means",
        "(A)",
    ),
    (
        axes[1],
        gmm_labels,
        "Autoencoder + GMM",
        "(B)",
    ),
]:
    for cluster_index in sorted(
        np.unique(labels)
    ):
        mask = labels == cluster_index

        axis.scatter(
            embedding_pca[mask, 0],
            embedding_pca[mask, 1],
            s=10,
            alpha=0.45,
            color=cluster_colours[
                int(cluster_index)
            ],
            edgecolors="none",
            rasterized=True,
            label=f"Cluster {cluster_index}",
        )

    axis.set_title(title)

    axis.set_xlabel(
        f"PC1 ({pc1_variance:.1f}% variance)"
    )

    axis.text(
        0.025,
        0.965,
        panel,
        transform=axis.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
    )

    axis.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

axes[0].set_ylabel(
    f"PC2 ({pc2_variance:.1f}% variance)"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "autoencoder_kmeans_gmm_pca.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "autoencoder_kmeans_gmm_pca.pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ==========================================================
# 16. FINAL SUMMARY
# ==========================================================

print("\n==============================================")
print("CONTROLLED AUTOENCODER BASELINE FINISHED")
print("==============================================")
print(f"Particles: {EXPECTED_N_IMAGES}")
print(f"Input size: {IMG_SIZE} x {IMG_SIZE}")
print(f"Embedding dimension: {LATENT_DIM}")
print(f"Epochs: {EPOCHS}")
print(f"Output folder: {OUTPUT_DIR}")
print("\nKey files:")
print("  autoencoder_embeddings.npy")
print("  autoencoder_embedding_image_paths.csv")
print("  autoencoder_kmeans_K2_cluster_assignments.csv")
print("  autoencoder_gmm_K2_cluster_assignments.csv")
print("  autoencoder_pca_coordinates.csv")
print("  autoencoder_clustering_metrics_K2.csv")
print("  autoencoder_kmeans_gmm_pca.png/pdf")