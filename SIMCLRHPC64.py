# ==========================================================
# UNSUPERVISED PARTICLE CLASSIFICATION — HPC VERSION
# 1024x1024 centred particle -> contour crop -> 224x224
# SimCLR with retained 64-D embedding + discarded 32-D projection
# GMM + t-SNE visualisation
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

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# ==========================================================
# 1. SETTINGS — HPC VERSION
# ==========================================================

IMAGE_DIR = Path("/mnt/hpccs01/home/n11555581/Code/CLASSIFICATIONFOLDER2")
OUTPUT_DIR = Path("/mnt/hpccs01/home/n11555581/Code/SIMCLR_CLASSIFICATION_TRIAL_EMBEDDING_64")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original CSV containing exactly the 7,415 particles used in the first trial.
ORIGINAL_ASSIGNMENTS_FILE = Path(
    "/mnt/hpccs01/home/n11555581/Code/"
    "SIMCLR_CLASSIFICATION_TRIAL/"
    "simclr_particle_cluster_assignments.csv"
)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100

# Retained representation used for K-means/GMM/PCA/t-SNE.
EMBEDDING_DIM = 64

# Discarded SimCLR projection used only by the contrastive loss.
PROJECTION_HIDDEN_DIM = 64
PROJECTION_OUTPUT_DIM = 32

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
    def __init__(
        self,
        image_dir,
        allowed_image_names,
        transform=None,
        base_transform=None
    ):
        all_image_paths = []

        for ext in [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.tif",
            "*.tiff",
            "*.bmp"
        ]:
            all_image_paths.extend(
                glob.glob(
                    os.path.join(
                        str(image_dir),
                        ext
                    )
                )
            )

        # Match by filename because the original CSV paths may refer to the
        # same files through a different mounted path.
        path_by_name = {}

        duplicate_names = set()

        for path in sorted(all_image_paths):
            image_name = os.path.basename(path)

            if image_name in path_by_name:
                duplicate_names.add(image_name)
            else:
                path_by_name[image_name] = path

        if duplicate_names:
            raise ValueError(
                "Duplicate filenames were found in IMAGE_DIR. "
                "Filename-based filtering would be ambiguous. "
                f"Examples: {sorted(duplicate_names)[:10]}"
            )

        requested_names = list(allowed_image_names)
        requested_name_set = set(requested_names)

        self.image_paths = [
            path_by_name[name]
            for name in requested_names
            if name in path_by_name
        ]

        missing_names = [
            name
            for name in requested_names
            if name not in path_by_name
        ]

        extra_names = [
            name
            for name in path_by_name
            if name not in requested_name_set
        ]

        audit_df = pd.DataFrame({
            "image_name": requested_names,
            "found_in_current_folder": [
                name in path_by_name
                for name in requested_names
            ],
            "resolved_image_path": [
                path_by_name.get(name, "")
                for name in requested_names
            ]
        })

        audit_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "selected_7415_dataset_audit.csv"
            ),
            index=False
        )

        pd.DataFrame({
            "extra_image_name": sorted(extra_names)
        }).to_csv(
            os.path.join(
                OUTPUT_DIR,
                "images_excluded_from_7415_trial.csv"
            ),
            index=False
        )

        if missing_names:
            pd.DataFrame({
                "missing_image_name": missing_names
            }).to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    "missing_original_7415_images.csv"
                ),
                index=False
            )

            raise FileNotFoundError(
                f"{len(missing_names)} of the original selected images "
                "were not found in the current folder. See "
                "missing_original_7415_images.csv."
            )

        if len(self.image_paths) != len(requested_names):
            raise RuntimeError(
                "The selected dataset size does not match the original "
                "assignment list."
            )

        self.transform = transform
        self.base_transform = base_transform

        print(
            f"Images available in folder: {len(all_image_paths)}",
            flush=True
        )
        print(
            f"Images selected for this trial: {len(self.image_paths)}",
            flush=True
        )
        print(
            f"Images excluded from this trial: {len(extra_names)}",
            flush=True
        )

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

if not ORIGINAL_ASSIGNMENTS_FILE.exists():
    raise FileNotFoundError(
        "The original assignment CSV was not found: "
        f"{ORIGINAL_ASSIGNMENTS_FILE}"
    )

original_assignments_df = pd.read_csv(
    ORIGINAL_ASSIGNMENTS_FILE
)

if "image_name" not in original_assignments_df.columns:
    raise KeyError(
        "The original assignment CSV must contain an 'image_name' column."
    )

selected_image_names = (
    original_assignments_df["image_name"]
    .astype(str)
    .str.strip()
    .tolist()
)

if len(selected_image_names) != len(set(selected_image_names)):
    duplicated_selected_names = (
        pd.Series(selected_image_names)
        .value_counts()
    )

    duplicated_selected_names = (
        duplicated_selected_names[
            duplicated_selected_names > 1
        ]
    )

    raise ValueError(
        "The original assignment CSV contains duplicate image names. "
        f"Examples: {duplicated_selected_names.head(10).to_dict()}"
    )

EXPECTED_ORIGINAL_COUNT = 7415

if len(selected_image_names) != EXPECTED_ORIGINAL_COUNT:
    raise ValueError(
        "The original assignment CSV does not contain exactly "
        f"{EXPECTED_ORIGINAL_COUNT} unique images. It contains "
        f"{len(selected_image_names)}."
    )

dataset = SimCLRParticleDataset(
    IMAGE_DIR,
    allowed_image_names=selected_image_names,
    transform=simclr_transform,
    base_transform=base_transform
)

if len(dataset) != EXPECTED_ORIGINAL_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_ORIGINAL_COUNT} selected particles, "
        f"but obtained {len(dataset)}."
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
    def __init__(
        self,
        embedding_dim=64,
        projection_hidden_dim=64,
        projection_output_dim=32
    ):
        super().__init__()

        self.encoder = models.resnet18(
            weights=None
        )

        self.encoder.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        backbone_output_dim = (
            self.encoder.fc.in_features
        )

        self.encoder.fc = nn.Identity()

        # Retained embedding head:
        # ResNet18 output 512 -> final representation 64.
        # This 64-D tensor is saved and used for downstream clustering.
        self.embedding_head = nn.Sequential(
            nn.Linear(
                backbone_output_dim,
                embedding_dim
            ),
            nn.ReLU()
        )

        # Discarded SimCLR projection head:
        # retained 64-D embedding -> hidden 64 -> projection 32.
        # Only z is used by the NT-Xent training objective.
        self.projector = nn.Sequential(
            nn.Linear(
                embedding_dim,
                projection_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                projection_hidden_dim,
                projection_output_dim
            )
        )

    def forward(self, x):
        backbone_features = self.encoder(x)

        h = self.embedding_head(
            backbone_features
        )

        z = self.projector(h)

        z = F.normalize(
            z,
            dim=1
        )

        return h, z


model = SimCLRModel(
    embedding_dim=EMBEDDING_DIM,
    projection_hidden_dim=PROJECTION_HIDDEN_DIM,
    projection_output_dim=PROJECTION_OUTPUT_DIM
).to(DEVICE)

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
    os.path.join(OUTPUT_DIR, "simclr_embedding64_projection32_model.pth")
)

plt.figure(figsize=(6, 4))
plt.plot(loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("SimCLR loss")
plt.title("SimCLR training loss")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "simclr_embedding64_training_loss.png"), dpi=300)
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

np.save(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embeddings_64.npy"
    ),
    embeddings
)

if embeddings.shape != (
    EXPECTED_ORIGINAL_COUNT,
    EMBEDDING_DIM
):
    raise ValueError(
        "Unexpected embedding matrix shape. Expected "
        f"({EXPECTED_ORIGINAL_COUNT}, {EMBEDDING_DIM}), "
        f"received {embeddings.shape}."
    )

embedding_manifest_df = pd.DataFrame({
    "embedding_row": np.arange(
        len(image_paths)
    ),
    "image_path": image_paths,
    "image_name": [
        os.path.basename(path)
        for path in image_paths
    ]
})

embedding_manifest_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embeddings_64_manifest.csv"
    ),
    index=False
)

print(
    "Confirmed retained embedding dimension:",
    embeddings.shape[1],
    flush=True
)


# ==========================================================
# 9. GMM MODEL SELECTION
# ==========================================================

cluster_metrics = []

for k in K_RANGE:
    print(
        f"Testing GMM k = {k}",
        flush=True
    )

    gmm_test = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42,
        n_init=50,
        max_iter=500,
        reg_covar=1e-6
    )

    labels_test = gmm_test.fit_predict(
        embeddings
    )

    sil = silhouette_score(
        embeddings,
        labels_test
    )

    db = davies_bouldin_score(
        embeddings,
        labels_test
    )

    ch = calinski_harabasz_score(
        embeddings,
        labels_test
    )

    bic = gmm_test.bic(
        embeddings
    )

    aic = gmm_test.aic(
        embeddings
    )

    cluster_metrics.append({
        "k": k,
        "bic": bic,
        "aic": aic,
        "silhouette_score": sil,
        "davies_bouldin_score": db,
        "calinski_harabasz_score": ch,
        "converged": gmm_test.converged_,
        "n_iterations": gmm_test.n_iter_
    })

metrics_df = pd.DataFrame(
    cluster_metrics
)

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embedding64_gmm_cluster_metrics.csv"
    ),
    index=False
)

print(
    metrics_df,
    flush=True
)


# ==========================================================
# 10. FINAL GMM
# ==========================================================

gmm_final = GaussianMixture(
    n_components=FINAL_K,
    covariance_type="diag",
    random_state=42,
    n_init=50,
    max_iter=500,
    reg_covar=1e-6
)

final_labels = gmm_final.fit_predict(
    embeddings
)

cluster_probabilities = gmm_final.predict_proba(
    embeddings
)

gmm_confidence = cluster_probabilities.max(
    axis=1
)

log_likelihood = gmm_final.score_samples(
    embeddings
)

results_df = pd.DataFrame({
    "image_path": image_paths,
    "image_name": [
        os.path.basename(path)
        for path in image_paths
    ],
    "gmm_cluster": final_labels,
    "gmm_confidence": gmm_confidence,
    "log_likelihood": log_likelihood
})

for cluster_number in range(FINAL_K):
    results_df[
        f"posterior_cluster_{cluster_number}"
    ] = cluster_probabilities[
        :,
        cluster_number
    ]

    results_df[
        f"posterior_cluster_{cluster_number}_percent"
    ] = (
        cluster_probabilities[
            :,
            cluster_number
        ]
        * 100
    )

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embedding64_gmm_particle_cluster_assignments.csv"
    ),
    index=False
)

results_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embedding64_gmm_particle_cluster_assignments.xlsx"
    ),
    index=False
)

print(
    "\nFinal GMM cluster counts:",
    flush=True
)

print(
    pd.Series(
        final_labels
    ).value_counts().sort_index(),
    flush=True
)

print(
    "\nFinal GMM component weights:",
    gmm_final.weights_,
    flush=True
)

print(
    "Final GMM converged:",
    gmm_final.converged_,
    flush=True
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
    "gmm_cluster": final_labels,
    "gmm_confidence": gmm_confidence,
    "log_likelihood": log_likelihood,
    "image_path": image_paths
})

tsne_df.to_csv(
    os.path.join(OUTPUT_DIR, "simclr_embedding64_gmm_tsne_clusters.csv"),
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
        label=f"GMM cluster {cl}"
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("SimCLR 64-D retained embedding + GMM")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "simclr_embedding64_tsne_gmm_clusters.png"),
    dpi=300
)
plt.close()


# ==========================================================
# 12. SAVE REPRESENTATIVE PARTICLES PER GMM COMPONENT
# ==========================================================

REP_DIR = os.path.join(
    OUTPUT_DIR,
    "simclr_embedding64_gmm_representative_particles"
)

os.makedirs(
    REP_DIR,
    exist_ok=True
)

for cluster_number in sorted(
    np.unique(final_labels)
):
    cluster_indices = np.where(
        final_labels == cluster_number
    )[0]

    cluster_mean = gmm_final.means_[
        cluster_number
    ]

    cluster_variance = gmm_final.covariances_[
        cluster_number
    ]

    # Mahalanobis distance is appropriate for diagonal-covariance GMMs.
    mahalanobis_squared = np.sum(
        (
            (
                embeddings[cluster_indices]
                - cluster_mean
            ) ** 2
        )
        / cluster_variance,
        axis=1
    )

    nearest_indices = cluster_indices[
        np.argsort(
            mahalanobis_squared
        )[:20]
    ]

    cluster_folder = os.path.join(
        REP_DIR,
        f"cluster_{cluster_number}"
    )

    os.makedirs(
        cluster_folder,
        exist_ok=True
    )

    for rank, index in enumerate(
        nearest_indices,
        start=1
    ):
        original = Image.open(
            image_paths[index]
        ).convert("L")

        normalised = normalize_particle_size_pil(
            original,
            output_size=IMG_SIZE,
            padding=5,
            fill_fraction=0.82
        )

        save_path = os.path.join(
            cluster_folder,
            (
                f"rank_{rank}_"
                f"{os.path.basename(image_paths[index])}"
            )
        )

        normalised.save(
            save_path
        )


# ==========================================================
# 13. GMM CONFIDENCE AND LOG-LIKELIHOOD HISTOGRAMS
# ==========================================================

plt.figure(figsize=(7, 5))

plt.hist(
    gmm_confidence,
    bins=50,
    edgecolor="black"
)

plt.xlabel(
    "Maximum posterior probability"
)

plt.ylabel(
    "Number of particles"
)

plt.title(
    f"GMM assignment confidence, K={FINAL_K}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embedding64_gmm_confidence_histogram.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(7, 5))

plt.hist(
    log_likelihood,
    bins=50,
    edgecolor="black"
)

plt.xlabel(
    "Per-particle GMM log-likelihood"
)

plt.ylabel(
    "Number of particles"
)

plt.title(
    f"GMM log-likelihood distribution, K={FINAL_K}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "simclr_embedding64_gmm_log_likelihood_histogram.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Done.",
    flush=True
)

print(
    "Results saved in:",
    os.path.abspath(OUTPUT_DIR),
    flush=True
)