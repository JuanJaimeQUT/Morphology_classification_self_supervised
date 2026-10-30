# ==========================================================
# UNSUPERVISED PARTICLE CLASSIFICATION — NO SIZE NORMALISATION
#
# Original centred 1024×1024 image
# -> direct resize to 224×224
# -> SimCLR
# -> K-means
# -> t-SNE visualisation
#
# This version preserves relative particle-size differences.
# ==========================================================

import os
import glob
import shutil
import random
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
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ==========================================================
# 1. SETTINGS — HPC VERSION
# ==========================================================

IMAGE_DIR = Path(
    "/mnt/hpccs01/home/n11555581/Code/CLASSIFICATIONFOLDER2"
)

OUTPUT_DIR = Path(
    "/mnt/hpccs01/home/n11555581/Code/"
    "SIMCLR_CLASSIFICATION_TRIAL_NO_NORMALISATION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
FEATURE_DIM = 128
LEARNING_RATE = 1e-4
TEMPERATURE = 0.5

K_RANGE = range(2, 11)
FINAL_K = 2
TSNE_PERPLEXITY = 30

RANDOM_SEED = 42
NUM_WORKERS = 4
N_REPRESENTATIVES = 20

# ==========================================================
# TRAIN ONLY ON ORIGINAL 7415 PARTICLES
# ==========================================================

TRAINING_LIST = Path(
    "/mnt/hpccs01/home/n11555581/Code/"
    "SIMCLR_CLASSIFICATION_TRIAL/"
    "SIM_PCA_KMEANS_TWO_CLASSES/"
    "kmeans_K2_cluster_assignments.csv"
)

training_df = pd.read_csv(TRAINING_LIST)

# Column containing the filenames
VALID_IMAGES = set(training_df["image_name"].astype(str))

print(f"Training images listed: {len(VALID_IMAGES)}")

# ==========================================================
# 2. REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================================
# 3. DEVICE
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Image folder:",
    IMAGE_DIR,
    flush=True
)

print(
    "Results folder:",
    OUTPUT_DIR,
    flush=True
)

print(
    "Using device:",
    DEVICE,
    flush=True
)

if DEVICE == "cuda":

    print(
        "GPU name:",
        torch.cuda.get_device_name(0),
        flush=True
    )


# ==========================================================
# 4. DIRECT IMAGE RESIZING
# ==========================================================
# IMPORTANT:
# No contour detection, cropping, scaling or centring is used.
#
# The complete original 1024×1024 image is resized directly
# to 224×224. Therefore, the relative particle size within
# the original canvas remains available to the network.
# ==========================================================

def resize_original_image(
    img,
    output_size=224
):
    """
    Resize the complete original image directly to a square
    output size without contour-based size normalisation.

    Parameters
    ----------
    img : PIL.Image.Image
        Original grayscale particle image.

    output_size : int
        Final image width and height.

    Returns
    -------
    PIL.Image.Image
        Directly resized grayscale image.
    """

    return img.resize(
        (output_size, output_size),
        Image.Resampling.BILINEAR
    )


# ==========================================================
# 5. DATASET FOR SIMCLR
# ==========================================================

class SimCLRParticleDataset(Dataset):

    def __init__(
        self,
        image_dir,
        transform=None,
        base_transform=None
    ):

        self.image_paths = []

        image_extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.tif",
            "*.tiff",
            "*.bmp"
        ]

        for ext in image_extensions:

            self.image_paths.extend(
                glob.glob(
                    os.path.join(
                        str(image_dir),
                        ext
                    )
                )
            )

        self.image_paths = [

            p for p in self.image_paths

            if os.path.basename(p) in VALID_IMAGES

        ]

        self.image_paths = sorted(self.image_paths)

        print(
            f"Using {len(self.image_paths)} images "
            f"from the original training set."
        )

        if len(self.image_paths) == 0:

            raise ValueError(
                "No images were found. "
                "Check the IMAGE_DIR path."
            )

        self.transform = transform
        self.base_transform = base_transform

        print(
            f"Found {len(self.image_paths)} images",
            flush=True
        )

    def __len__(self):

        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        idx
    ):

        path = self.image_paths[idx]

        # Open as grayscale
        img = Image.open(
            path
        ).convert("L")

        # Directly resize the complete image.
        # No contour-based normalization is performed.
        img = resize_original_image(
            img,
            output_size=IMG_SIZE
        )

        # ResNet-18 receives three channels.
        # These are three identical grayscale channels.
        img = img.convert("RGB")

        # Two independent augmented views for SimCLR
        xi = self.transform(
            img
        )

        xj = self.transform(
            img
        )

        # Non-augmented image for feature extraction
        x_base = self.base_transform(
            img
        )

        return (
            xi,
            xj,
            x_base,
            path
        )


# ==========================================================
# 6. SIMCLR AUGMENTATIONS
# ==========================================================

simclr_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=180
    ),

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


# ==========================================================
# 7. CREATE DATASET AND LOADERS
# ==========================================================

dataset = SimCLRParticleDataset(
    IMAGE_DIR,
    transform=simclr_transform,
    base_transform=base_transform
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
    pin_memory=(
        DEVICE == "cuda"
    )
)


feature_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=(
        DEVICE == "cuda"
    )
)


print(
    "Number of training batches:",
    len(loader),
    flush=True
)

print(
    "Number of feature batches:",
    len(feature_loader),
    flush=True
)


# ==========================================================
# 8. SAVE DIRECT-RESIZE CHECK IMAGE
# ==========================================================
# This figure confirms that the entire original image is
# simply resized. Particle-size differences remain visible.
# ==========================================================

debug_paths = dataset.image_paths[:8]

plt.figure(
    figsize=(12, 4)
)

for i, path in enumerate(
    debug_paths
):

    original = Image.open(
        path
    ).convert("L")

    resized = resize_original_image(
        original,
        output_size=IMG_SIZE
    )

    # Original image
    plt.subplot(
        2,
        len(debug_paths),
        i + 1
    )

    plt.imshow(
        original,
        cmap="gray"
    )

    plt.axis("off")

    if i == 0:

        plt.ylabel(
            "Original"
        )

    # Directly resized image
    plt.subplot(
        2,
        len(debug_paths),
        i + 1 + len(debug_paths)
    )

    plt.imshow(
        resized,
        cmap="gray"
    )

    plt.axis("off")

    if i == 0:

        plt.ylabel(
            "Direct resize\n224×224"
        )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "direct_resize_check.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 9. SIMCLR MODEL
# ==========================================================

class SimCLRModel(nn.Module):

    def __init__(
        self,
        feature_dim=128
    ):

        super().__init__()

        # Randomly initialized ResNet-18
        self.encoder = models.resnet18(
            weights=None
        )

        # Three-channel input:
        # each channel contains the same grayscale values.
        self.encoder.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_features = (
            self.encoder.fc.in_features
        )

        # Remove original classification layer
        self.encoder.fc = nn.Identity()

        # SimCLR projection head
        self.projector = nn.Sequential(

            nn.Linear(
                num_features,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                feature_dim
            )
        )

    def forward(
        self,
        x
    ):

        # 512-dimensional encoder representation
        h = self.encoder(
            x
        )

        # 128-dimensional projection
        z = self.projector(
            h
        )

        # L2 normalization for contrastive learning
        z = F.normalize(
            z,
            dim=1
        )

        return h, z


model = SimCLRModel(
    feature_dim=FEATURE_DIM
).to(DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# 10. NT-XENT LOSS
# ==========================================================

def nt_xent_loss(
    z_i,
    z_j,
    temperature=0.5
):

    batch_size = z_i.shape[0]

    # Combine both augmented views
    z = torch.cat(
        [z_i, z_j],
        dim=0
    )

    z = F.normalize(
        z,
        dim=1
    )

    # Pairwise cosine similarity
    similarity = (
        torch.matmul(
            z,
            z.T
        )
        / temperature
    )

    # Exclude self-similarity
    mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z.device
    )

    similarity = similarity.masked_fill(
        mask,
        -9e15
    )

    # Positive pairs
    positives = torch.cat([

        torch.diag(
            similarity,
            batch_size
        ),

        torch.diag(
            similarity,
            -batch_size
        )

    ], dim=0)

    denominator = torch.logsumexp(
        similarity,
        dim=1
    )

    loss = (
        -positives
        + denominator
    )

    return loss.mean()


# ==========================================================
# 11. TRAIN SIMCLR
# ==========================================================

loss_history = []


for epoch in range(
    EPOCHS
):

    model.train()

    epoch_loss = 0.0
    processed_samples = 0

    for batch_idx, (
        xi,
        xj,
        _,
        _
    ) in enumerate(loader):

        xi = xi.to(
            DEVICE,
            non_blocking=True
        )

        xj = xj.to(
            DEVICE,
            non_blocking=True
        )

        _, zi = model(
            xi
        )

        _, zj = model(
            xj
        )

        loss = nt_xent_loss(
            zi,
            zj,
            temperature=TEMPERATURE
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        current_batch_size = xi.size(0)

        epoch_loss += (
            loss.item()
            * current_batch_size
        )

        processed_samples += (
            current_batch_size
        )

        if batch_idx % 10 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.6f}",
                flush=True
            )

    # Use the number of processed samples because
    # drop_last=True excludes the final incomplete batch.
    epoch_loss /= processed_samples

    loss_history.append(
        epoch_loss
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Mean SimCLR loss: "
        f"{epoch_loss:.6f}",
        flush=True
    )


# ==========================================================
# 12. SAVE TRAINED MODEL
# ==========================================================

model_path = (
    OUTPUT_DIR
    / "simclr_model_no_normalisation.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print(
    "Saved model to:",
    model_path,
    flush=True
)


# ==========================================================
# 13. SAVE TRAINING LOSS
# ==========================================================

loss_df = pd.DataFrame({
    "epoch": np.arange(
        1,
        EPOCHS + 1
    ),
    "simclr_loss": loss_history
})

loss_df.to_csv(
    OUTPUT_DIR
    / "simclr_training_loss_no_normalisation.csv",
    index=False
)


plt.figure(
    figsize=(6, 4)
)

plt.plot(
    np.arange(
        1,
        EPOCHS + 1
    ),
    loss_history,
    marker="o",
    markersize=3
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "SimCLR loss"
)

plt.title(
    "SimCLR training loss — no size normalization"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "simclr_training_loss_no_normalisation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 14. EXTRACT 512-DIMENSIONAL EMBEDDINGS
# ==========================================================

model.eval()

embeddings = []
image_paths = []


with torch.no_grad():

    for batch_idx, (
        _,
        _,
        x_base,
        paths
    ) in enumerate(feature_loader):

        x_base = x_base.to(
            DEVICE,
            non_blocking=True
        )

        h, _ = model(
            x_base
        )

        embeddings.append(
            h.cpu().numpy()
        )

        image_paths.extend(
            list(paths)
        )

        if batch_idx % 10 == 0:

            print(
                "Extracting embeddings: "
                f"batch {batch_idx + 1}/"
                f"{len(feature_loader)}",
                flush=True
            )


embeddings = np.vstack(
    embeddings
)


print(
    "Embedding matrix shape:",
    embeddings.shape,
    flush=True
)


if embeddings.shape[0] != len(
    image_paths
):

    raise ValueError(
        "The number of embeddings does not match "
        "the number of image paths."
    )


np.save(
    OUTPUT_DIR
    / "simclr_embeddings_no_normalisation.npy",
    embeddings
)


embedding_assignments_df = pd.DataFrame({

    "embedding_index": np.arange(
        embeddings.shape[0]
    ),

    "image_path": image_paths,

    "image_name": [
        os.path.basename(path)
        for path in image_paths
    ]

})


embedding_assignments_df.to_csv(
    OUTPUT_DIR
    / "simclr_embedding_image_paths_no_normalisation.csv",
    index=False
)


# ==========================================================
# 15. K-MEANS MODEL-SELECTION METRICS
# ==========================================================

cluster_metrics = []


for k in K_RANGE:

    print(
        f"Testing k = {k}",
        flush=True
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=50
    )

    labels_k = kmeans.fit_predict(
        embeddings
    )

    silhouette = silhouette_score(
        embeddings,
        labels_k
    )

    davies_bouldin = davies_bouldin_score(
        embeddings,
        labels_k
    )

    calinski_harabasz = (
        calinski_harabasz_score(
            embeddings,
            labels_k
        )
    )

    cluster_metrics.append({

        "k": k,

        "silhouette_score":
            silhouette,

        "davies_bouldin_score":
            davies_bouldin,

        "calinski_harabasz_score":
            calinski_harabasz
    })


metrics_df = pd.DataFrame(
    cluster_metrics
)


metrics_df.to_csv(
    OUTPUT_DIR
    / "simclr_kmeans_cluster_metrics_no_normalisation.csv",
    index=False
)


print(
    "\nK-means model-selection metrics:"
)

print(
    metrics_df.to_string(
        index=False
    ),
    flush=True
)


# ==========================================================
# 16. SAVE CLUSTER-METRIC PLOTS
# ==========================================================

metric_plot_settings = [

    (
        "silhouette_score",
        "Silhouette coefficient",
        "Higher is better"
    ),

    (
        "davies_bouldin_score",
        "Davies–Bouldin index",
        "Lower is better"
    ),

    (
        "calinski_harabasz_score",
        "Calinski–Harabasz index",
        "Higher is better"
    )
]


for (
    metric_column,
    y_label,
    interpretation
) in metric_plot_settings:

    plt.figure(
        figsize=(6, 4)
    )

    plt.plot(
        metrics_df["k"],
        metrics_df[metric_column],
        marker="o",
        linewidth=1.5
    )

    plt.xlabel(
        "Number of clusters, K"
    )

    plt.ylabel(
        y_label
    )

    plt.title(
        f"{y_label}\n{interpretation}"
    )

    plt.xticks(
        list(K_RANGE)
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / (
            f"no_normalisation_"
            f"{metric_column}_vs_k.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# 17. FINAL K-MEANS CLUSTERING
# ==========================================================

kmeans_final = KMeans(
    n_clusters=FINAL_K,
    random_state=RANDOM_SEED,
    n_init=50
)


final_labels = kmeans_final.fit_predict(
    embeddings
)


results_df = pd.DataFrame({

    "image_path": image_paths,

    "image_name": [
        os.path.basename(path)
        for path in image_paths
    ],

    "cluster": final_labels

})


results_df.to_csv(
    OUTPUT_DIR
    / "simclr_particle_cluster_assignments_no_normalisation.csv",
    index=False
)


print(
    "\nFinal cluster counts:"
)

print(
    results_df[
        "cluster"
    ].value_counts().sort_index(),
    flush=True
)


# ==========================================================
# 18. t-SNE VISUALISATION
# ==========================================================

print(
    "Running t-SNE...",
    flush=True
)


n_samples = embeddings.shape[0]

effective_perplexity = min(
    TSNE_PERPLEXITY,
    max(
        5,
        (n_samples - 1) // 3
    )
)


print(
    "Using t-SNE perplexity:",
    effective_perplexity,
    flush=True
)


tsne = TSNE(
    n_components=2,
    perplexity=effective_perplexity,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_SEED
)


emb_tsne = tsne.fit_transform(
    embeddings
)


tsne_df = pd.DataFrame({

    "TSNE1": emb_tsne[:, 0],

    "TSNE2": emb_tsne[:, 1],

    "cluster": final_labels,

    "image_path": image_paths,

    "image_name": [
        os.path.basename(path)
        for path in image_paths
    ]

})


tsne_df.to_csv(
    OUTPUT_DIR
    / "simclr_tsne_clusters_no_normalisation.csv",
    index=False
)


plt.figure(
    figsize=(7, 6)
)


for cluster in sorted(
    np.unique(final_labels)
):

    cluster_mask = (
        final_labels
        == cluster
    )

    plt.scatter(
        emb_tsne[
            cluster_mask,
            0
        ],
        emb_tsne[
            cluster_mask,
            1
        ],
        s=12,
        alpha=0.7,
        label=f"Cluster {cluster}"
    )


plt.xlabel(
    "t-SNE 1"
)

plt.ylabel(
    "t-SNE 2"
)

plt.title(
    "SimCLR embedding space without size normalization"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "simclr_tsne_kmeans_no_normalisation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 19. PCA VISUALISATION
# ==========================================================

from sklearn.decomposition import PCA


pca = PCA(
    n_components=2
)


emb_pca = pca.fit_transform(
    embeddings
)


pca_df = pd.DataFrame({

    "PC1": emb_pca[:, 0],

    "PC2": emb_pca[:, 1],

    "cluster": final_labels,

    "image_path": image_paths,

    "image_name": [
        os.path.basename(path)
        for path in image_paths
    ]

})


pca_df.to_csv(
    OUTPUT_DIR
    / "simclr_pca_clusters_no_normalisation.csv",
    index=False
)


plt.figure(
    figsize=(7, 6)
)


for cluster in sorted(
    np.unique(final_labels)
):

    cluster_mask = (
        final_labels
        == cluster
    )

    plt.scatter(
        emb_pca[
            cluster_mask,
            0
        ],
        emb_pca[
            cluster_mask,
            1
        ],
        s=12,
        alpha=0.7,
        label=f"Cluster {cluster}"
    )


plt.xlabel(
    "PC1 "
    f"({pca.explained_variance_ratio_[0] * 100:.1f}%)"
)

plt.ylabel(
    "PC2 "
    f"({pca.explained_variance_ratio_[1] * 100:.1f}%)"
)

plt.title(
    "SimCLR PCA space without size normalization"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "simclr_pca_kmeans_no_normalisation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 20. SAVE REPRESENTATIVE PARTICLES PER CLUSTER
# ==========================================================

REP_DIR = (
    OUTPUT_DIR
    / "simclr_representative_particles_no_normalisation"
)

REP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for cluster in sorted(
    np.unique(final_labels)
):

    cluster_idx = np.where(
        final_labels
        == cluster
    )[0]

    cluster_center = (
        kmeans_final.cluster_centers_[
            cluster
        ]
    )

    distances = np.linalg.norm(
        embeddings[cluster_idx]
        - cluster_center,
        axis=1
    )

    nearest_idx = cluster_idx[
        np.argsort(
            distances
        )[
            :N_REPRESENTATIVES
        ]
    ]

    cluster_folder = (
        REP_DIR
        / f"cluster_{cluster}"
    )

    cluster_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Cluster {cluster}: saving "
        f"{len(nearest_idx)} "
        "representative particles",
        flush=True
    )

    for rank, idx in enumerate(
        nearest_idx,
        start=1
    ):

        source_path = image_paths[idx]

        if not os.path.exists(
            source_path
        ):

            print(
                "WARNING: image not found:",
                source_path,
                flush=True
            )

            continue

        original = Image.open(
            source_path
        ).convert("L")

        # Directly resize the original complete image.
        # No contour-based normalization is used.
        resized = resize_original_image(
            original,
            output_size=IMG_SIZE
        )

        save_name = (
            f"rank_{rank}_"
            f"{os.path.basename(source_path)}"
        )

        save_path = (
            cluster_folder
            / save_name
        )

        resized.save(
            save_path
        )


# ==========================================================
# 21. SAVE RUN CONFIGURATION
# ==========================================================

configuration_df = pd.DataFrame({

    "parameter": [
        "image_directory",
        "output_directory",
        "image_size",
        "batch_size",
        "epochs",
        "feature_dimension",
        "encoder_dimension",
        "learning_rate",
        "temperature",
        "final_k",
        "tsne_perplexity",
        "random_seed",
        "size_normalisation"
    ],

    "value": [
        str(IMAGE_DIR),
        str(OUTPUT_DIR),
        IMG_SIZE,
        BATCH_SIZE,
        EPOCHS,
        FEATURE_DIM,
        512,
        LEARNING_RATE,
        TEMPERATURE,
        FINAL_K,
        effective_perplexity,
        RANDOM_SEED,
        "No — direct resize only"
    ]

})


configuration_df.to_csv(
    OUTPUT_DIR
    / "run_configuration_no_normalisation.csv",
    index=False
)


# ==========================================================
# 22. FINAL SUMMARY
# ==========================================================

print(
    "\n==============================================="
)

print(
    "SIMCLR WITHOUT SIZE NORMALISATION FINISHED"
)

print(
    "==============================================="
)

print(
    "Number of particles:",
    embeddings.shape[0]
)

print(
    "Embedding dimension:",
    embeddings.shape[1]
)

print(
    "Final number of clusters:",
    FINAL_K
)

print(
    "Size normalization:",
    "NOT APPLIED"
)

print(
    "Input processing:",
    "complete image directly resized to 224×224"
)

print(
    "Results saved in:",
    os.path.abspath(
        OUTPUT_DIR
    ),
    flush=True
)