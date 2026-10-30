# ==========================================================
# UNSUPERVISED PARTICLE CLASSIFICATION — HPC VERSION
# 1024x1024 centred particle -> contour crop -> 224x224
# SimCLR + K-means + t-SNE visualisation
# ==========================================================

import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# ==========================================================
# 1. SETTINGS — HPC VERSION
# ==========================================================

IMAGE_DIR = Path("/mnt/hpccs01/home/n11555581/Code/CLASSIFICATIONFOLDER2")
OUTPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/SIMCLR_CLASSIFICATION_TRIAL")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
FEATURE_DIM = 128
LEARNING_RATE = 1e-4
TEMPERATURE = 0.5

K_RANGE = range(2, 11)
FINAL_K = 2
TSNE_PERPLEXITY = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Image folder:", IMAGE_DIR, flush=True)
print("Results folder:", OUTPUT_DIR, flush=True)
print("Using device:", DEVICE, flush=True)

if DEVICE == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0), flush=True)


# ==========================================================
# 2. CONTOUR-BASED SIZE NORMALISATION
# ==========================================================

def normalize_particle_size_pil(img, output_size=224, padding=5, fill_fraction=0.82):
    arr = np.array(img)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(arr, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img.resize((output_size, output_size))

    img_h, img_w = arr.shape
    centre_x, centre_y = img_w / 2, img_h / 2

    best_contour = None
    best_score = -np.inf

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        cnt_cx = x + w / 2
        cnt_cy = y + h / 2

        dist = np.sqrt(
            (cnt_cx - centre_x) ** 2 +
            (cnt_cy - centre_y) ** 2
        )

        score = area - 2.0 * dist

        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return img.resize((output_size, output_size))

    x, y, w, h = cv2.boundingRect(best_contour)

    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, arr.shape[1])
    y1 = min(y + h + padding, arr.shape[0])

    cropped = arr[y0:y1, x0:x1]

    crop_h, crop_w = cropped.shape
    scale = (output_size * fill_fraction) / max(crop_w, crop_h)

    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return Image.fromarray(canvas)


# ==========================================================
# 3. DATASET FOR SIMCLR
# ==========================================================

class SimCLRParticleDataset(Dataset):
    def __init__(self, image_dir, transform=None, base_transform=None):
        self.image_paths = []

        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError("No images found. Check IMAGE_DIR path.")

        self.transform = transform
        self.base_transform = base_transform

        print(f"Found {len(self.image_paths)} images", flush=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        img = Image.open(path).convert("L")

        img = normalize_particle_size_pil(
            img,
            output_size=IMG_SIZE,
            padding=5,
            fill_fraction=0.82
        )

        img = img.convert("RGB")

        xi = self.transform(img)
        xj = self.transform(img)

        x_base = self.base_transform(img)

        return xi, xj, x_base, path


simclr_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.RandomResizedCrop(
        size=IMG_SIZE,
        scale=(0.85, 1.0),
        ratio=(0.95, 1.05)
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor()
])

base_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SimCLRParticleDataset(
    IMAGE_DIR,
    transform=simclr_transform,
    base_transform=base_transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True if DEVICE == "cuda" else False
)

feature_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    pin_memory=True if DEVICE == "cuda" else False
)


# ==========================================================
# 4. SAVE NORMALISATION CHECK IMAGE
# ==========================================================

debug_paths = dataset.image_paths[:8]

plt.figure(figsize=(12, 4))

for i, path in enumerate(debug_paths):
    original = Image.open(path).convert("L")
    normalised = normalize_particle_size_pil(original, IMG_SIZE)

    plt.subplot(2, len(debug_paths), i + 1)
    plt.imshow(original, cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.ylabel("Original")

    plt.subplot(2, len(debug_paths), i + 1 + len(debug_paths))
    plt.imshow(normalised, cmap="gray")
    plt.axis("off")
    if i == 0:
        plt.ylabel("Normalised 224")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "size_normalisation_check.png"), dpi=300)
plt.close()


# ==========================================================
# 5. SIMCLR MODEL
# ==========================================================

class SimCLRModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.encoder = models.resnet18(weights=None)

        self.encoder.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


model = SimCLRModel(feature_dim=FEATURE_DIM).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ==========================================================
# 6. NT-XENT LOSS
# ==========================================================

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]

    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity = similarity.masked_fill(mask, -9e15)

    positives = torch.cat([
        torch.diag(similarity, batch_size),
        torch.diag(similarity, -batch_size)
    ], dim=0)

    denominator = torch.logsumexp(similarity, dim=1)

    loss = -positives + denominator

    return loss.mean()


# ==========================================================
# 7. TRAIN SIMCLR
# ==========================================================

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (xi, xj, _, _) in enumerate(loader):
        xi = xi.to(DEVICE)
        xj = xj.to(DEVICE)

        _, zi = model(xi)
        _, zj = model(xj)

        loss = nt_xent_loss(zi, zj, temperature=TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xi.size(0)

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.6f}",
                flush=True
            )

    epoch_loss /= len(loader.dataset)
    loss_history.append(epoch_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] Mean SimCLR loss: {epoch_loss:.6f}",
        flush=True
    )

torch.save(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "simclr_model.pth")
)

plt.figure(figsize=(6, 4))
plt.plot(loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("SimCLR loss")
plt.title("SimCLR training loss")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "simclr_training_loss.png"), dpi=300)
plt.close()


# ==========================================================
# 8. EXTRACT EMBEDDINGS
# ==========================================================

model.eval()

embeddings = []
image_paths = []

with torch.no_grad():
    for batch_idx, (_, _, x_base, paths) in enumerate(feature_loader):
        x_base = x_base.to(DEVICE)

        h, _ = model(x_base)

        embeddings.append(h.cpu().numpy())
        image_paths.extend(paths)

        if batch_idx % 10 == 0:
            print(
                f"Extracting embeddings: batch {batch_idx+1}/{len(feature_loader)}",
                flush=True
            )

embeddings = np.vstack(embeddings)

print("Embedding matrix:", embeddings.shape, flush=True)

np.save(os.path.join(OUTPUT_DIR, "simclr_embeddings.npy"), embeddings)


# ==========================================================
# 9. K-MEANS CLUSTERING
# ==========================================================

cluster_metrics = []

for k in K_RANGE:
    print(f"Testing k = {k}", flush=True)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=50
    )

    labels = kmeans.fit_predict(embeddings)

    sil = silhouette_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    ch = calinski_harabasz_score(embeddings, labels)

    cluster_metrics.append({
        "k": k,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "calinski_harabasz_score": ch
    })

metrics_df = pd.DataFrame(cluster_metrics)
metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "simclr_kmeans_cluster_metrics.csv"),
    index=False
)

print(metrics_df, flush=True)


# ==========================================================
# 10. FINAL K-MEANS
# ==========================================================

kmeans_final = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=50
)

final_labels = kmeans_final.fit_predict(embeddings)

results_df = pd.DataFrame({
    "image_path": image_paths,
    "image_name": [os.path.basename(p) for p in image_paths],
    "cluster": final_labels
})

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "simclr_particle_cluster_assignments.csv"),
    index=False
)


# ==========================================================
# 11. t-SNE VISUALISATION
# ==========================================================

print("Running t-SNE...", flush=True)

tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    learning_rate="auto",
    init="pca",
    random_state=42
)

emb_tsne = tsne.fit_transform(embeddings)

tsne_df = pd.DataFrame({
    "TSNE1": emb_tsne[:, 0],
    "TSNE2": emb_tsne[:, 1],
    "cluster": final_labels,
    "image_path": image_paths
})

tsne_df.to_csv(
    os.path.join(OUTPUT_DIR, "simclr_tsne_clusters.csv"),
    index=False
)

plt.figure(figsize=(7, 6))

for cl in sorted(np.unique(final_labels)):
    idx = final_labels == cl

    plt.scatter(
        emb_tsne[idx, 0],
        emb_tsne[idx, 1],
        s=12,
        alpha=0.7,
        label=f"Cluster {cl}"
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("SimCLR embedding space + K-means")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "simclr_tsne_kmeans_clusters.png"),
    dpi=300
)
plt.close()


# ==========================================================
# 12. SAVE REPRESENTATIVE PARTICLES PER CLUSTER
# ==========================================================

REP_DIR = os.path.join(OUTPUT_DIR, "simclr_representative_particles")
os.makedirs(REP_DIR, exist_ok=True)

for cl in sorted(np.unique(final_labels)):
    cluster_idx = np.where(final_labels == cl)[0]
    cluster_center = kmeans_final.cluster_centers_[cl]

    distances = np.linalg.norm(
        embeddings[cluster_idx] - cluster_center,
        axis=1
    )

    nearest_idx = cluster_idx[np.argsort(distances)[:20]]

    cluster_folder = os.path.join(REP_DIR, f"cluster_{cl}")
    os.makedirs(cluster_folder, exist_ok=True)

    for rank, idx in enumerate(nearest_idx):
        original = Image.open(image_paths[idx]).convert("L")

        normalised = normalize_particle_size_pil(
            original,
            output_size=IMG_SIZE,
            padding=5,
            fill_fraction=0.82
        )

        save_path = os.path.join(
            cluster_folder,
            f"rank_{rank+1}_{os.path.basename(image_paths[idx])}"
        )

        normalised.save(save_path)


print("Done.", flush=True)
print("Results saved in:", os.path.abspath(OUTPUT_DIR), flush=True)