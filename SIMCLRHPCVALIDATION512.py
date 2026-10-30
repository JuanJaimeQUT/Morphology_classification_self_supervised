import os
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from scipy.spatial.distance import jensenshannon

from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


# ==========================================================
# PATHS
# ==========================================================

OUTPUT_DIR = "/mnt/hpccs01/home/n11555581/Code/SIMCLR_CLASSIFICATION_TRIAL"

EMBEDDINGS_FILE = os.path.join(
    OUTPUT_DIR,
    "simclr_embeddings.npy"
)

ASSIGNMENTS_FILE = os.path.join(
    OUTPUT_DIR,
    "simclr_particle_cluster_assignments.csv"
)

NEW_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "SIM_TSNE_GMM_THREE_CLUSTERS_FINAL_VALIDATE"
)

os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)


# ==========================================================
# UNSEEN-IMAGE VALIDATION CONFIGURATION
# ==========================================================
# IMPORTANT:
# 1. Set SIMCLR_MODEL_FILE to the trained SimCLR model used to create
#    simclr_embeddings.npy.
# 2. The image preprocessing below MUST be identical to the preprocessing
#    used when the original 7,415 embeddings were generated.
# 3. The loader supports a TorchScript model or a complete torch.nn.Module.
#    If your checkpoint contains only a state_dict, reconstruct the exact
#    architecture in load_simclr_model() and then load that state_dict.

ALL_IMAGES_DIR = "/mnt/hpccs01/home/n11555581/Code/CLASSIFICATIONFOLDER2"

SIMCLR_MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "simclr_model.pth"
)

VALID_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}

BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
IMAGE_SIZE = 224

# Set this to True only when the saved model returns the projection-head
# representation used to fit the GMM. If the saved model returns several
# outputs, adjust extract_model_embedding() below.
L2_NORMALIZE_EXTRACTED_EMBEDDINGS = False


# ==========================================================
# MANUAL MORPHOLOGY GROUPS
# ==========================================================

GROUP_1 = [
    "M_SOP02030828_particle_1_padded.png",
    "M_SOP02030767_particle_2_padded.png",
    "M_SOP02031224_particle_1_padded.png",
    "M_SOP02031255_particle_1_padded.png",
    "M_SOP02031393_particle_1_padded.png",
    "M_SOP02033258_particle_1_padded.png",
    "M_SOP02033137_particle_1_padded.png",
    "M_SOP02033722_particle_1_padded.png",
    "M_SOP02034248_particle_1_padded.png",
    "M_SOP02042208_particle_1_padded.png",
    "M_SOP02043093_particle_1_padded.png",
    "M_SOP02043299_particle_1_padded.png",
    "M_SOP02043800_particle_1_padded.png",
    "M_SOP02050701_particle_2_padded.png",
    "M_SOP02051773_particle_1_padded.png",
    "M_SOP02052040_particle_1_padded.png",
    "M_SOP02052098_particle_1_padded.png",
    "M_SOP02052395_particle_1_padded.png",
    "M_SOP02052697_particle_2_padded.png",
    "M_SOP02052938_particle_1_padded.png",
    "M_SOP02053253_particle_1_padded.png",
    "M_SOP02052739_particle_2_padded.png",
    "M_SOP02053504_particle_1_padded.png",
    "M_SOP02063305_particle_1_padded.png",
    "M_SOP02070639_particle_1_padded.png",
]

GROUP_2 = [
    "M_SOP02030178_particle_2_padded.png",
    "M_SOP02030940_particle_1_padded.png",
    "M_SOP02031109_particle_2_padded.png",
    "M_SOP02031722_particle_1_padded.png",
    "M_SOP02031813_particle_1_padded.png",
    "M_SOP02031855_particle_1_padded.png",
    "M_SOP02032012_particle_2_padded.png",
    "M_SOP02032839_particle_1_padded.png",
    "M_SOP02032882_particle_1_padded.png",
    "M_SOP02033245_particle_1_padded.png",
    "M_SOP02033360_particle_1_padded.png",
    "M_SOP02033436_particle_1_padded.png",
    "M_SOP02033899_particle_1_padded.png",
    "M_SOP02041406_particle_1_padded.png",
    "M_SOP02041923_particle_1_padded.png",
    "M_SOP02042339_particle_1_padded.png",
    "M_SOP02043082_particle_2_padded.png",
    "M_SOP02043093_particle_2_padded.png",
    "M_SOP02051691_particle_1_padded.png",
    "M_SOP02051788_particle_1_padded.png",
    "M_SOP02052218_particle_1_padded.png",
    "M_SOP02052744_particle_1_padded.png",
    "M_SOP02053007_particle_1_padded.png",
    "M_SOP02060466_particle_1_padded.png",
    "M_SOP02060464_particle_1_padded.png",
]

GROUP_3 = [
    "M_SOP02030018_particle_1_padded.png",
    "M_SOP02030214_particle_1_padded.png",
    "M_SOP02030851_particle_1_padded.png",
    "M_SOP02030908_particle_1_padded.png",
    "M_SOP02031441_particle_1_padded.png",
    "M_SOP02031575_particle_1_padded.png",
    "M_SOP02032067_particle_1_padded.png",
    "M_SOP02032798_particle_1_padded.png",
    "M_SOP02040572_particle_1_padded.png",
    "M_SOP02040838_particle_1_padded.png",
    "M_SOP02040853_particle_1_padded.png",
    "M_SOP02043343_particle_1_padded.png",
    "M_SOP02043559_particle_2_padded.png",
    "M_SOP02051228_particle_1_padded.png",
    "M_SOP02051485_particle_1_padded.png",
    "M_SOP02051479_particle_1_padded.png",
    "M_SOP02061616_particle_1_padded.png",
    "M_SOP02070515_particle_1_padded.png",
    "M_SOP02071365_particle_1_padded.png",
    "M_SOP02072053_particle_1_padded.png",
    "M_SOP02072653_particle_1_padded.png",
    "M_SOP02073607_particle_1_padded.png",
    "M_SOP02073712_particle_1_padded.png",
    "M_SOP02030497_particle_1_padded.png",
    "M_SOP02030026_particle_1_padded.png",
]

GROUP_4 = [
    "M_SOP02031381_particle_1_padded.png",
    "M_SOP02040163_particle_1_padded.png",
    "M_SOP02040484_particle_1_padded.png",
    "M_SOP02041057_particle_1_padded.png",
    "M_SOP02042388_particle_1_padded.png",
    "M_SOP02042759_particle_1_padded.png",
    "M_SOP02042853_particle_2_padded.png",
    "M_SOP02042884_particle_2_padded.png",
    "M_SOP02042947_particle_1_padded.png",
    "M_SOP02043111_particle_1_padded.png",
    "M_SOP02043567_particle_1_padded.png",
    "M_SOP02051147_particle_1_padded.png",
    "M_SOP02051463_particle_2_padded.png",
    "M_SOP02051649_particle_1_padded.png",
    "M_SOP02052302_particle_1_padded.png",
    "M_SOP02053697_particle_2_padded.png",
    "M_SOP02053401_particle_1_padded.png",
    "M_SOP02053546_particle_1_padded.png",
    "M_SOP02053662_particle_1_padded.png",
]

GROUP_5 = [
    "M_SOP02030744_particle_1_padded.png",
    "M_SOP02030768_particle_1_padded.png",
    "M_SOP02030998_particle_2_padded.png",
    "M_SOP02031522_particle_1_padded.png",
    "M_SOP02031525_particle_1_padded.png",
    "M_SOP02031661_particle_1_padded.png",
    "M_SOP02031817_particle_1_padded.png",
    "M_SOP02031853_particle_1_padded.png",
    "M_SOP02032038_particle_1_padded.png",
    "M_SOP02033046_particle_1_padded.png",
    "M_SOP02034191_particle_1_padded.png",
    "M_SOP02034208_particle_1_padded.png",
    "M_SOP02034513_particle_1_padded.png",
    "M_SOP02034811_particle_1_padded.png",
    "M_SOP02034858_particle_1_padded.png",
    "M_SOP02043924_particle_1_padded.png",
    "M_SOP02041956_particle_1_padded.png",
    "M_SOP02052064_particle_1_padded.png",
    "M_SOP02060073_particle_1_padded.png",
    "M_SOP02060089_particle_1_padded.png",
    "M_SOP02072778_particle_1_padded.png",
    "M_SOP02073060_particle_1_padded.png",
    "M_SOP02031129_particle_1_padded.png",
    "M_SOP02063931_particle_1_padded.png",
    "M_SOP02042492_particle_1_padded.png",
]

GROUP_6 = [
    "M_SOP02030215_particle_1_padded.png",
    "M_SOP02030275_particle_1_padded.png",
    "M_SOP02031324_particle_1_padded.png",
    "M_SOP02033409_particle_1_padded.png",
    "M_SOP02033702_particle_1_padded.png",
    "M_SOP02033894_particle_1_padded.png",
    "M_SOP02034692_particle_1_padded.png",
    "M_SOP02040036_particle_1_padded.png",
    "M_SOP02040855_particle_1_padded.png",
    "M_SOP02041130_particle_1_padded.png",
    "M_SOP02041796_particle_1_padded.png",
    "M_SOP02041895_particle_1_padded.png",
    "M_SOP02041338_particle_1_padded.png",
    "M_SOP02042092_particle_1_padded.png",
    "M_SOP02042093_particle_1_padded.png",
    "M_SOP02042125_particle_1_padded.png",
    "M_SOP02042156_particle_1_padded.png",
    "M_SOP02042107_particle_1_padded.png",
    "M_SOP02042193_particle_1_padded.png",
    "M_SOP02042269_particle_1_padded.png",
    "M_SOP02042503_particle_1_padded.png",
    "M_SOP02042562_particle_1_padded.png",
    "M_SOP02042603_particle_1_padded.png",
    "M_SOP02042657_particle_1_padded.png",
    "M_SOP02043103_particle_2_padded.png",
    "M_SOP02043158_particle_1_padded.png",
    "M_SOP02043163_particle_1_padded.png",
    "M_SOP02043277_particle_1_padded.png",
    "M_SOP02043282_particle_1_padded.png",
    "M_SOP02043279_particle_1_padded.png",
    "M_SOP02043289_particle_1_padded.png",
    "M_SOP02043304_particle_1_padded.png",
    "M_SOP02043744_particle_1_padded.png",
    "M_SOP02043905_particle_1_padded.png",
    "M_SOP02050199_particle_1_padded.png",
    "M_SOP02050276_particle_1_padded.png",
    "M_SOP02050458_particle_1_padded.png",
    "M_SOP02050482_particle_1_padded.png",
    "M_SOP02050615_particle_1_padded.png",
    "M_SOP02050742_particle_1_padded.png",
    "M_SOP02050863_particle_1_padded.png",
    "M_SOP02051075_particle_1_padded.png",
    "M_SOP02051144_particle_1_padded.png",
    "M_SOP02051116_particle_1_padded.png",
    "M_SOP02051563_particle_1_padded.png",
    "M_SOP02051706_particle_2_padded.png",
    "M_SOP02051787_particle_1_padded.png",
    "M_SOP02051825_particle_1_padded.png",
    "M_SOP02051847_particle_1_padded.png",
    "M_SOP02051864_particle_1_padded.png",
    "M_SOP02051956_particle_1_padded.png",
    "M_SOP02052437_particle_1_padded.png",
    "M_SOP02052450_particle_1_padded.png",
    "M_SOP02052532_particle_2_padded.png",
    "M_SOP02052534_particle_1_padded.png",
    "M_SOP02052575_particle_1_padded.png",
    "M_SOP02052656_particle_1_padded.png",
    "M_SOP02052617_particle_1_padded.png",
    "M_SOP02052640_particle_1_padded.png",
    "M_SOP02052707_particle_1_padded.png",
    "M_SOP02052742_particle_1_padded.png",
    "M_SOP02052748_particle_1_padded.png",
    "M_SOP02072550_particle_1_padded.png",
    "M_SOP02052957_particle_1_padded.png",
    "M_SOP02053353_particle_1_padded.png",
    "M_SOP02053298_particle_1_padded.png",
    "M_SOP02053625_particle_2_padded.png",
    "M_SOP02053678_particle_2_padded.png",
    "M_SOP02060492_particle_2_padded.png",
    "M_SOP02063410_particle_1_padded.png",
    "M_SOP02063543_particle_1_padded.png",
    "M_SOP02063621_particle_2_padded.png",
    "M_SOP02063855_particle_1_padded.png",
    "M_SOP02070523_particle_1_padded.png",
    "M_SOP02071003_particle_1_padded.png",
    "M_SOP02071642_particle_1_padded.png",
    "M_SOP02071672_particle_2_padded.png",
    "M_SOP02072343_particle_1_padded.png",
    "M_SOP02073125_particle_1_padded.png",
    "M_SOP02073349_particle_1_padded.png",
    "M_SOP02073687_particle_1_padded.png",
    ""


]


# ==========================================================
# LOAD DATA
# ==========================================================

embeddings = np.load(EMBEDDINGS_FILE)

assignments_df = pd.read_csv(
    ASSIGNMENTS_FILE
)

image_paths = assignments_df["image_path"].tolist()
image_names = assignments_df["image_name"].tolist()

if embeddings.ndim != 2:
    raise ValueError(
        "Expected a two-dimensional embedding matrix, "
        f"but received shape {embeddings.shape}."
    )

if embeddings.shape[1] != 512:
    raise ValueError(
        "This validation script is configured for the original "
        "512-dimensional SimCLR encoder representation, but the "
        f"loaded embedding matrix has shape {embeddings.shape}."
    )

if embeddings.shape[0] != len(image_paths):
    raise ValueError(
        "The number of embeddings does not match the number "
        "of image paths in the assignments file."
    )

name_to_index = {
    name: idx
    for idx, name in enumerate(image_names)
}

print("Embeddings shape:", embeddings.shape, flush=True)
print("Number of image paths:", len(image_paths), flush=True)


# ==========================================================
# GMM CLUSTERING
# ==========================================================

N_CLUSTERS = 3

if N_CLUSTERS == 2:

    manual_groups = {
        "Radial": GROUP_1 + GROUP_2 + GROUP_3 + GROUP_5,
        "Mosaic": GROUP_4 + GROUP_6
    }

else:

    manual_groups = {
        "Manual class 1": GROUP_1,
        "Manual class 2": GROUP_2,
        "Manual class 3": GROUP_3,
        "Manual class 4": GROUP_4,
        "Manual class 5": GROUP_5,
        "Manual class 6": GROUP_6,
    }

gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="diag",
    random_state=42,
    n_init=50,
    max_iter=500,
    reg_covar=1e-6
)

labels = gmm.fit_predict(embeddings)

cluster_probabilities = gmm.predict_proba(
    embeddings
)

#ll = gmm.score_samples(embeddings)

gmm_confidence = cluster_probabilities.max(axis=1)


# ==========================================================
# SCORE IMAGES NOT USED TO TRAIN THE SIMCLR/GMM PIPELINE
# ==========================================================

# Exact model and preprocessing definitions copied from the original
# SimCLR training script. These must not be changed for validation.

def normalize_particle_size_pil(
    img,
    output_size=224,
    padding=5,
    fill_fraction=0.82
):
    arr = np.array(img)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    blurred = cv2.GaussianBlur(
        arr,
        (5, 5),
        0
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img.resize(
            (output_size, output_size)
        )

    img_h, img_w = arr.shape
    centre_x = img_w / 2
    centre_y = img_h / 2

    best_contour = None
    best_score = -np.inf

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        x, y, width, height = cv2.boundingRect(
            contour
        )

        contour_centre_x = x + width / 2
        contour_centre_y = y + height / 2

        distance = np.sqrt(
            (contour_centre_x - centre_x) ** 2
            + (contour_centre_y - centre_y) ** 2
        )

        score = area - 2.0 * distance

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return img.resize(
            (output_size, output_size)
        )

    x, y, width, height = cv2.boundingRect(
        best_contour
    )

    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(
        x + width + padding,
        arr.shape[1]
    )
    y1 = min(
        y + height + padding,
        arr.shape[0]
    )

    cropped = arr[y0:y1, x0:x1]

    crop_height, crop_width = cropped.shape

    scale = (
        output_size * fill_fraction
    ) / max(crop_width, crop_height)

    new_width = max(
        1,
        int(crop_width * scale)
    )
    new_height = max(
        1,
        int(crop_height * scale)
    )

    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (output_size, output_size),
        dtype=np.uint8
    )

    x_offset = (
        output_size - new_width
    ) // 2

    y_offset = (
        output_size - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return Image.fromarray(canvas)


class SimCLRModel(nn.Module):
    def __init__(self, feature_dim=128):
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

        number_of_features = (
            self.encoder.fc.in_features
        )

        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(
                number_of_features,
                512
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                feature_dim
            )
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(
            z,
            dim=1
        )

        return h, z


class ParticleImageDataset(Dataset):
    def __init__(
        self,
        image_paths_to_load,
        transform
    ):
        self.image_paths = list(
            image_paths_to_load
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[
            index
        ]

        try:
            with Image.open(image_path) as image:
                # This exactly matches the original feature-extraction path:
                # grayscale -> contour size normalization -> RGB -> ToTensor.
                image = image.convert("L")

                image = normalize_particle_size_pil(
                    image,
                    output_size=IMAGE_SIZE,
                    padding=5,
                    fill_fraction=0.82
                )

                image = image.convert("RGB")
                image_tensor = self.transform(
                    image
                )

            return image_tensor, image_path, ""

        except Exception as error:
            placeholder = torch.zeros(
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                dtype=torch.float32
            )

            return (
                placeholder,
                image_path,
                str(error)
            )


def load_simclr_model(
    model_file,
    device
):
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            "The trained SimCLR state_dict was not found: "
            f"{model_file}"
        )

    model = SimCLRModel(
        feature_dim=128
    ).to(device)

    state_dict = torch.load(
        model_file,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print(
        "Loaded the original 512-D SimCLR ResNet18 "
        "state_dict successfully.",
        flush=True
    )

    return model


def extract_model_embedding(
    model_output
):
    # The original script saved h from:
    # h, _ = model(x_base)
    # Therefore validation must also use h, not z.
    if not isinstance(
        model_output,
        (tuple, list)
    ):
        raise TypeError(
            "Expected the SimCLR model to return "
            "(encoder_features, projection)."
        )

    embedding = model_output[0]

    if embedding.ndim != 2:
        raise ValueError(
            "Expected encoder embedding shape "
            "[batch, features], but received "
            f"{tuple(embedding.shape)}."
        )

    return embedding


def find_all_particle_images(root_directory):
    if not os.path.isdir(root_directory):
        raise NotADirectoryError(
            f"ALL_IMAGES_DIR does not exist or is not accessible: "
            f"{root_directory}"
        )

    found_paths = []

    for current_root, _, filenames in os.walk(root_directory):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            if extension in VALID_IMAGE_EXTENSIONS:
                found_paths.append(
                    os.path.join(current_root, filename)
                )

    return sorted(found_paths)


def create_embedding_transform():
    # Exact base_transform from the original embedding extraction.
    # Do not add ImageNet normalization or random augmentation here.
    return transforms.Compose([
        transforms.ToTensor()
    ])


def generate_unseen_embeddings(
    model,
    unseen_paths,
    device
):
    dataset = ParticleImageDataset(
        unseen_paths,
        transform=create_embedding_transform()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda"
    )

    embedding_batches = []
    valid_paths = []
    failed_rows = []

    model.eval()

    with torch.no_grad():
        for batch_number, (
            image_batch,
            batch_paths,
            batch_errors
        ) in enumerate(dataloader, start=1):

            valid_mask = np.array([
                error == ""
                for error in batch_errors
            ])

            for path_value, error_value in zip(
                batch_paths,
                batch_errors
            ):
                if error_value != "":
                    failed_rows.append({
                        "image_path": path_value,
                        "error": error_value
                    })

            if not np.any(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            valid_tensor = image_batch[
                torch.as_tensor(
                    valid_indices,
                    dtype=torch.long
                )
            ].to(
                device,
                non_blocking=True
            )

            model_output = model(valid_tensor)
            batch_embeddings = extract_model_embedding(
                model_output
            )

            if L2_NORMALIZE_EXTRACTED_EMBEDDINGS:
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings,
                    p=2,
                    dim=1
                )

            embedding_batches.append(
                batch_embeddings.detach().cpu().numpy()
            )

            valid_paths.extend([
                batch_paths[index]
                for index in valid_indices
            ])

            if batch_number % 20 == 0:
                print(
                    f"Processed {batch_number} unseen-image batches.",
                    flush=True
                )

    if len(embedding_batches) == 0:
        raise RuntimeError(
            "No valid unseen images could be embedded."
        )

    unseen_embedding_array = np.concatenate(
        embedding_batches,
        axis=0
    )

    failed_df = pd.DataFrame(
        failed_rows,
        columns=["image_path", "error"]
    )

    return unseen_embedding_array, valid_paths, failed_df


print("\n===============================================")
print("UNSEEN-IMAGE GENERALISATION TEST")
print("===============================================\n")

all_available_image_paths = find_all_particle_images(
    ALL_IMAGES_DIR
)

training_image_names = {
    str(name).strip().lower()
    for name in image_names
}

unseen_image_paths = [
    path
    for path in all_available_image_paths
    if os.path.basename(path).strip().lower()
    not in training_image_names
]

print(
    f"Images found in complete folder: "
    f"{len(all_available_image_paths)}",
    flush=True
)

print(
    f"Images represented in original embedding table: "
    f"{len(training_image_names)}",
    flush=True
)

print(
    f"Images identified as unseen: "
    f"{len(unseen_image_paths)}",
    flush=True
)

if len(unseen_image_paths) == 0:
    raise RuntimeError(
        "No unseen images were found. Check whether image_name in the "
        "assignment CSV matches the basenames in ALL_IMAGES_DIR."
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"Embedding unseen images on device: {device}",
    flush=True
)

simclr_model = load_simclr_model(
    SIMCLR_MODEL_FILE,
    device
)

unseen_embeddings, valid_unseen_paths, failed_unseen_df = (
    generate_unseen_embeddings(
        simclr_model,
        unseen_image_paths,
        device
    )
)

if unseen_embeddings.shape[1] != embeddings.shape[1]:
    raise ValueError(
        "The unseen embedding dimension does not match the original "
        "embedding dimension.\n"
        f"Original embeddings: {embeddings.shape[1]} features\n"
        f"Unseen embeddings: {unseen_embeddings.shape[1]} features\n"
        "This means the wrong model output or preprocessing is being used. "
        "Adjust extract_model_embedding() so it returns the exact SimCLR "
        "representation used to fit the GMM."
    )

np.save(
    os.path.join(
        NEW_OUTPUT_DIR,
        "unseen_simclr_embeddings.npy"
    ),
    unseen_embeddings
)

failed_unseen_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "unseen_images_failed_to_load.csv"
    ),
    index=False
)

unseen_probabilities = gmm.predict_proba(
    unseen_embeddings
)

unseen_labels = gmm.predict(
    unseen_embeddings
)

unseen_confidence = unseen_probabilities.max(
    axis=1
)

# ==========================================================
# TRAINING VS UNSEEN POSTERIOR-CONFIDENCE SUMMARY
# ==========================================================

training_max_posterior = cluster_probabilities.max(axis=1)
unseen_max_posterior = unseen_probabilities.max(axis=1)

# Posterior entropy:
# 0 = highly certain assignment
# 1 = maximum uncertainty across all clusters
training_probabilities_safe = np.clip(
    cluster_probabilities,
    1e-300,
    1.0
)

unseen_probabilities_safe = np.clip(
    unseen_probabilities,
    1e-300,
    1.0
)

training_entropy = -np.sum(
    training_probabilities_safe
    * np.log(training_probabilities_safe),
    axis=1
)

unseen_entropy = -np.sum(
    unseen_probabilities_safe
    * np.log(unseen_probabilities_safe),
    axis=1
)

training_normalized_entropy = (
    training_entropy / np.log(N_CLUSTERS)
)

unseen_normalized_entropy = (
    unseen_entropy / np.log(N_CLUSTERS)
)

posterior_thresholds = [
    0.90,
    0.95,
    0.99
]


def posterior_summary_row(
    dataset_name,
    maximum_posteriors,
    normalized_entropies
):
    row = {
        "dataset": dataset_name,
        "n_particles": len(maximum_posteriors),
        "mean_maximum_posterior":
            np.mean(maximum_posteriors),
        "median_maximum_posterior":
            np.median(maximum_posteriors),
        "minimum_maximum_posterior":
            np.min(maximum_posteriors),
        "mean_normalized_posterior_entropy":
            np.mean(normalized_entropies),
        "median_normalized_posterior_entropy":
            np.median(normalized_entropies)
    }

    for threshold in posterior_thresholds:

        threshold_label = str(
            threshold
        ).replace(".", "_")

        count = int(
            np.sum(
                maximum_posteriors >= threshold
            )
        )

        percentage = (
            100
            * count
            / len(maximum_posteriors)
        )

        row[
            f"n_posterior_ge_{threshold_label}"
        ] = count

        row[
            f"percent_posterior_ge_{threshold_label}"
        ] = percentage

    return row


posterior_generalization_summary_df = pd.DataFrame([
    posterior_summary_row(
        dataset_name="training",
        maximum_posteriors=training_max_posterior,
        normalized_entropies=training_normalized_entropy
    ),
    posterior_summary_row(
        dataset_name="independent_validation",
        maximum_posteriors=unseen_max_posterior,
        normalized_entropies=unseen_normalized_entropy
    )
])

posterior_generalization_summary_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_validation_posterior_summary.csv"
    ),
    index=False
)

posterior_generalization_summary_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_validation_posterior_summary.xlsx"
    ),
    index=False
)

print(
    "\nTraining versus validation posterior summary:",
    flush=True
)

print(
    posterior_generalization_summary_df.to_string(
        index=False
    ),
    flush=True
)
unseen_log_likelihood = gmm.score_samples(
    unseen_embeddings
)

training_log_likelihood = gmm.score_samples(
    embeddings
)

unseen_results_df = pd.DataFrame({
    "dataset": "unseen_test",
    "image_path": valid_unseen_paths,
    "image_name": [
        os.path.basename(path)
        for path in valid_unseen_paths
    ],
    "gmm_cluster": unseen_labels,
    "gmm_confidence": unseen_confidence,
    "log_likelihood": unseen_log_likelihood
})

for cluster_number in range(N_CLUSTERS):
    unseen_results_df[
        f"posterior_cluster_{cluster_number}"
    ] = unseen_probabilities[:, cluster_number]

    unseen_results_df[
        f"posterior_cluster_{cluster_number}_percent"
    ] = (
        unseen_probabilities[:, cluster_number]
        * 100
    )

unseen_results_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "unseen_images_gmm_scores.csv"
    ),
    index=False
)

unseen_results_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "unseen_images_gmm_scores.xlsx"
    ),
    index=False
)

training_likelihood_df = pd.DataFrame({
    "dataset": "training",
    "image_path": image_paths,
    "image_name": image_names,
    "gmm_cluster": labels,
    "gmm_confidence": gmm_confidence,
    "log_likelihood": training_log_likelihood
})

combined_likelihood_df = pd.concat(
    [
        training_likelihood_df,
        unseen_results_df[[
            "dataset",
            "image_path",
            "image_name",
            "gmm_cluster",
            "gmm_confidence",
            "log_likelihood"
        ]]
    ],
    ignore_index=True
)

combined_likelihood_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_likelihoods.csv"
    ),
    index=False
)

# ==========================================================
# JOURNAL-QUALITY HISTOGRAM:
# TRAINING VS UNSEEN-PARTICLE LOG-LIKELIHOOD
# ==========================================================

# Use identical bin edges for a fair comparison.
combined_minimum = min(
    np.min(training_log_likelihood),
    np.min(unseen_log_likelihood)
)

combined_maximum = max(
    np.max(training_log_likelihood),
    np.max(unseen_log_likelihood)
)

shared_bins = np.linspace(
    combined_minimum,
    combined_maximum,
    51
)

# Publication-style font settings.
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0
})

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

# Muted, colourblind-friendly colours.
training_colour = "#4C72B0"
unseen_colour = "#DD8452"

ax.hist(
    training_log_likelihood,
    bins=shared_bins,
    density=True,
    alpha=0.45,
    color=training_colour,
    edgecolor="none",
    label=(
        f"Training dataset "
        f"(n = {len(training_log_likelihood):,})"
    )
)

ax.hist(
    unseen_log_likelihood,
    bins=shared_bins,
    density=True,
    alpha=0.55,
    color=unseen_colour,
    edgecolor="none",
    label=(
        f"Independent validation dataset "
        f"(n = {len(unseen_log_likelihood):,})"
    )
)

training_median = np.median(
    training_log_likelihood
)

unseen_median = np.median(
    unseen_log_likelihood
)

ax.axvline(
    training_median,
    color=training_colour,
    linestyle="--",
    linewidth=2.0,
    label=(
        f"Median (training) "    
    )
)

ax.axvline(
    unseen_median,
    color='#C44E52',
    linestyle=":",
    linewidth=2.4,
    label=(
        f"Median (validation) "
    )
)

ax.set_xlabel(
    "Per-particle GMM log-likelihood"
)

ax.set_ylabel(
    "Probability density"
)

# Remove the title because the information should appear
# in the journal figure caption.
ax.set_title("")

# Show all four axes (publication style).
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

ax.tick_params(
    axis="both",
    direction="out",
    length=5,
    width=1
)

ax.legend(
    loc="upper left",
    frameon=False,
    handlelength=2.8,
    borderaxespad=0.4
)

ax.margins(x=0.01)

fig.tight_layout()

# High-resolution raster version.
fig.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_log_likelihood_histogram.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

# Vector version preferred by many journals.
fig.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_log_likelihood_histogram.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)

# Empirical cumulative distribution plot is often easier to compare than
# overlaid histograms when sample sizes differ.
training_sorted = np.sort(
    training_log_likelihood
)

unseen_sorted = np.sort(
    unseen_log_likelihood
)

training_ecdf = np.arange(
    1,
    len(training_sorted) + 1
) / len(training_sorted)

unseen_ecdf = np.arange(
    1,
    len(unseen_sorted) + 1
) / len(unseen_sorted)

plt.figure(figsize=(8, 6))

plt.plot(
    training_sorted,
    training_ecdf,
    linewidth=2,
    label="Training"
)

plt.plot(
    unseen_sorted,
    unseen_ecdf,
    linewidth=2,
    label="Unseen"
)

plt.xlabel("Per-particle GMM log-likelihood")
plt.ylabel("Empirical cumulative probability")
plt.title("Training versus unseen log-likelihood ECDF")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_log_likelihood_ecdf.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Numerical distribution comparisons.
ks_result = ks_2samp(
    training_log_likelihood,
    unseen_log_likelihood,
    alternative="two-sided",
    mode="auto"
)

mw_result = mannwhitneyu(
    training_log_likelihood,
    unseen_log_likelihood,
    alternative="two-sided"
)

wasserstein_value = wasserstein_distance(
    training_log_likelihood,
    unseen_log_likelihood
)

training_q01 = np.quantile(
    training_log_likelihood,
    0.01
)

training_q05 = np.quantile(
    training_log_likelihood,
    0.05
)

unseen_below_training_q01 = 100 * np.mean(
    unseen_log_likelihood < training_q01
)

unseen_below_training_q05 = 100 * np.mean(
    unseen_log_likelihood < training_q05
)

likelihood_summary_df = pd.DataFrame([
    {
        "dataset": "training",
        "n_particles": len(training_log_likelihood),
        "mean_log_likelihood": np.mean(training_log_likelihood),
        "std_log_likelihood": np.std(
            training_log_likelihood,
            ddof=1
        ),
        "median_log_likelihood": np.median(training_log_likelihood),
        "q01_log_likelihood": np.quantile(
            training_log_likelihood,
            0.01
        ),
        "q05_log_likelihood": np.quantile(
            training_log_likelihood,
            0.05
        ),
        "q25_log_likelihood": np.quantile(
            training_log_likelihood,
            0.25
        ),
        "q75_log_likelihood": np.quantile(
            training_log_likelihood,
            0.75
        ),
        "q95_log_likelihood": np.quantile(
            training_log_likelihood,
            0.95
        ),
        "q99_log_likelihood": np.quantile(
            training_log_likelihood,
            0.99
        ),
        "mean_gmm_confidence": np.mean(gmm_confidence),
        "median_gmm_confidence": np.median(gmm_confidence)
    },
    {
        "dataset": "unseen_test",
        "n_particles": len(unseen_log_likelihood),
        "mean_log_likelihood": np.mean(unseen_log_likelihood),
        "std_log_likelihood": np.std(
            unseen_log_likelihood,
            ddof=1
        ),
        "median_log_likelihood": np.median(unseen_log_likelihood),
        "q01_log_likelihood": np.quantile(
            unseen_log_likelihood,
            0.01
        ),
        "q05_log_likelihood": np.quantile(
            unseen_log_likelihood,
            0.05
        ),
        "q25_log_likelihood": np.quantile(
            unseen_log_likelihood,
            0.25
        ),
        "q75_log_likelihood": np.quantile(
            unseen_log_likelihood,
            0.75
        ),
        "q95_log_likelihood": np.quantile(
            unseen_log_likelihood,
            0.95
        ),
        "q99_log_likelihood": np.quantile(
            unseen_log_likelihood,
            0.99
        ),
        "mean_gmm_confidence": np.mean(unseen_confidence),
        "median_gmm_confidence": np.median(unseen_confidence)
    }
])

likelihood_summary_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_likelihood_summary.csv"
    ),
    index=False
)

distribution_test_df = pd.DataFrame([{
    "ks_statistic": ks_result.statistic,
    "ks_p_value": ks_result.pvalue,
    "mann_whitney_u": mw_result.statistic,
    "mann_whitney_p_value": mw_result.pvalue,
    "wasserstein_distance": wasserstein_value,
    "training_1_percent_log_likelihood_threshold":
        training_q01,
    "training_5_percent_log_likelihood_threshold":
        training_q05,
    "unseen_percent_below_training_1_percent_threshold":
        unseen_below_training_q01,
    "unseen_percent_below_training_5_percent_threshold":
        unseen_below_training_q05
}])

distribution_test_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_distribution_tests.csv"
    ),
    index=False
)

cluster_comparison_rows = []

for cluster_number in range(N_CLUSTERS):
    train_cluster_values = training_log_likelihood[
        labels == cluster_number
    ]

    unseen_cluster_values = unseen_log_likelihood[
        unseen_labels == cluster_number
    ]

    cluster_comparison_rows.append({
        "cluster": cluster_number,
        "n_training": len(train_cluster_values),
        "n_unseen": len(unseen_cluster_values),
        "training_fraction_percent":
            100 * len(train_cluster_values) / len(labels),
        "unseen_fraction_percent":
            100 * len(unseen_cluster_values) / len(unseen_labels),
        "training_mean_log_likelihood":
            np.mean(train_cluster_values)
            if len(train_cluster_values) > 0
            else np.nan,
        "unseen_mean_log_likelihood":
            np.mean(unseen_cluster_values)
            if len(unseen_cluster_values) > 0
            else np.nan,
        "training_median_log_likelihood":
            np.median(train_cluster_values)
            if len(train_cluster_values) > 0
            else np.nan,
        "unseen_median_log_likelihood":
            np.median(unseen_cluster_values)
            if len(unseen_cluster_values) > 0
            else np.nan
    })

cluster_comparison_df = pd.DataFrame(
    cluster_comparison_rows
)

cluster_comparison_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "training_vs_unseen_cluster_comparison.csv"
    ),
    index=False
)

print("\nTraining/unseen likelihood summary:")
print(
    likelihood_summary_df.to_string(index=False),
    flush=True
)

print("\nDistribution comparison:")
print(
    distribution_test_df.to_string(index=False),
    flush=True
)

print("\nCluster proportions and likelihoods:")
print(
    cluster_comparison_df.to_string(index=False),
    flush=True
)

print(
    "\nUnseen-image scores and train/test comparison files saved.",
    flush=True
)


# ==========================================================
# SAVE CLUSTER ASSIGNMENTS
# ==========================================================

results_df = pd.DataFrame({
    "image_path": image_paths,
    "image_name": image_names,
    "gmm_cluster": labels,
    "gmm_confidence": gmm_confidence
})

results_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"gmm_K{N_CLUSTERS}_cluster_assignments.csv"
    ),
    index=False
)

print(
    results_df["gmm_cluster"].value_counts().sort_index(),
    flush=True
)


# ==========================================================
# t-SNE VISUALISATION
# ==========================================================

print("Running t-SNE...", flush=True)

n_samples = embeddings.shape[0]

tsne_perplexity = min(
    30,
    max(5, (n_samples - 1) // 3)
)

print(
    f"Using t-SNE perplexity: {tsne_perplexity}",
    flush=True
)

tsne = TSNE(
    n_components=2,
    perplexity=tsne_perplexity,
    learning_rate="auto",
    init="pca",
    random_state=42
)

emb_tsne = tsne.fit_transform(
    embeddings
)

tsne_df = pd.DataFrame({
    "TSNE1": emb_tsne[:, 0],
    "TSNE2": emb_tsne[:, 1],
    "gmm_cluster": labels,
    "gmm_confidence": gmm_confidence,
    "image_path": image_paths,
    "image_name": image_names
})

tsne_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"gmm_K{N_CLUSTERS}_tsne_coordinates.csv"
    ),
    index=False
)


# ==========================================================
# FIND MANUAL PARTICLE INDICES
# ==========================================================

manual_group_indices = {}

for group_name, group_files in manual_groups.items():

    indices = []

    for name in group_files:

        if name in name_to_index:
            indices.append(
                name_to_index[name]
            )

        else:
            print(
                f"WARNING: {name} not found in image list",
                flush=True
            )

    manual_group_indices[group_name] = indices


# ==========================================================
# PRINT MANUAL PARTICLES AND THEIR GMM CLUSTERS
# ==========================================================

for group_name, indices in manual_group_indices.items():

    print(
        f"\n{group_name}",
        flush=True
    )

    for idx in indices:

        print(
            image_names[idx],
            "-> GMM cluster",
            labels[idx],
            "| confidence:",
            round(gmm_confidence[idx], 4),
            flush=True
        )


# ==========================================================
# MANUAL CLASS vs GMM AGREEMENT TABLE
# ==========================================================

summary = []

for group_name, group_files in manual_groups.items():

    counts = np.zeros(
        N_CLUSTERS,
        dtype=int
    )

    found_count = 0
    missing_count = 0

    for name in group_files:

        if name in name_to_index:

            idx = name_to_index[name]

            counts[labels[idx]] += 1
            found_count += 1

        else:
            missing_count += 1

    if found_count > 0:

        dominant_cluster = int(
            np.argmax(counts)
        )

        purity = (
            100
            * counts[dominant_cluster]
            / found_count
        )

    else:

        dominant_cluster = None
        purity = np.nan

    row = {
        "manual_class": group_name,
        "n_found": found_count,
        "n_missing": missing_count,
        "dominant_gmm_cluster": dominant_cluster,
        "purity_percent": purity
    }

    for c in range(N_CLUSTERS):
        row[f"GMM_{c}"] = counts[c]

    summary.append(row)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"manual_vs_GMM_K{N_CLUSTERS}_summary.csv"
    ),
    index=False
)

print(
    "\nManual vs GMM summary:",
    flush=True
)

print(
    summary_df,
    flush=True
)


# ==========================================================
# PLOT 1: GMM CLUSTERS ONLY
# ==========================================================

plt.figure(figsize=(8, 6))

cluster_colors = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
]

for cl in sorted(np.unique(labels)):

    idx = labels == cl

    plt.scatter(
        emb_tsne[idx, 0],
        emb_tsne[idx, 1],
        s=12,
        alpha=0.7,
        color=cluster_colors[
            cl % len(cluster_colors)
        ],
        label=f"GMM cluster {cl}"
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.title(
    f"SimCLR embeddings + GMM clustering, K={N_CLUSTERS}"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"simclr_gmm_K{N_CLUSTERS}_tsne.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# PLOT 2: GMM CLUSTERS + MANUAL GROUPS
# ==========================================================

plt.figure(figsize=(9, 7))

for cl in sorted(np.unique(labels)):

    idx = labels == cl

    plt.scatter(
        emb_tsne[idx, 0],
        emb_tsne[idx, 1],
        s=10,
        alpha=0.18,
        color=cluster_colors[
            cl % len(cluster_colors)
        ],
        label=f"GMM cluster {cl}"
    )

if N_CLUSTERS == 2:

    manual_styles = {
        "Radial": {
            "color": "#0072B2",
            "marker": "o"
        },
        "Mosaic": {
            "color": "#D55E00",
            "marker": "s"
        },
    }

else:

    manual_styles = {
        "Manual class 1": {
            "color": "#0072B2",
            "marker": "o"
        },
        "Manual class 2": {
            "color": "#E69F00",
            "marker": "s"
        },
        "Manual class 3": {
            "color": "#009E73",
            "marker": "^"
        },
        "Manual class 4": {
            "color": "#D55E00",
            "marker": "D"
        },
        "Manual class 5": {
            "color": "#CC79A7",
            "marker": "P"
        },
        "Manual class 6": {
            "color": "#000000",
            "marker": "X"
        },
    }

for group_name, indices in manual_group_indices.items():

    if len(indices) == 0:
        continue

    style = manual_styles[group_name]

    plt.scatter(
        emb_tsne[indices, 0],
        emb_tsne[indices, 1],
        s=180,
        c=style["color"],
        marker=style["marker"],
        edgecolors="black",
        linewidth=1.2,
        label=group_name
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.title(
    "Manual morphology groups over "
    f"SimCLR GMM clusters, K={N_CLUSTERS}"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"simclr_gmm_K{N_CLUSTERS}_tsne_manual_groups.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# PCA VISUALISATIONS (analogous to the t-SNE figures)
# ==========================================================

print("Running PCA...", flush=True)

pca = PCA(n_components=2, random_state=42)
emb_pca = pca.fit_transform(embeddings)

pca_df = pd.DataFrame({
    "PC1": emb_pca[:,0],
    "PC2": emb_pca[:,1],
    "gmm_cluster": labels,
    "gmm_confidence": gmm_confidence,
    "image_path": image_paths,
    "image_name": image_names
})

pca_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"gmm_K{N_CLUSTERS}_pca_coordinates.csv"
    ),
    index=False
)

# -----------------------------
# PCA Plot 1: GMM clusters
# -----------------------------
plt.figure(figsize=(8,6))

for cl in sorted(np.unique(labels)):
    idx = labels == cl
    plt.scatter(
        emb_pca[idx,0],
        emb_pca[idx,1],
        s=12,
        alpha=0.7,
        color=cluster_colors[cl % len(cluster_colors)],
        label=f"GMM cluster {cl}"
    )

var = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({var[0]:.1f}%)")
plt.ylabel(f"PC2 ({var[1]:.1f}%)")
plt.title(f"SimCLR embeddings + GMM clustering (PCA), K={N_CLUSTERS}")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"simclr_gmm_K{N_CLUSTERS}_pca.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ==========================================================
# PCA PLOT — GMM CLUSTERS WITH EXPERT-LABEL OVERLAY
# JOURNAL-READY VERSION
# ==========================================================
# Background:
#   GMM cluster 0 = blue
#   GMM cluster 1 = orange
#
# Expert labels:
#   Radial = blue circles
#   Mosaic = orange squares
#
# The GMM population is shown using small transparent points,
# while expert-labelled particles are highlighted using
# smaller filled markers with white borders.
# ==========================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ----------------------------------------------------------
# COLOURS
# ----------------------------------------------------------
# These colours match the autoencoder–SimCLR comparison:
# cluster 0 = blue
# cluster 1 = orange.

cluster_colours = {
    0: "#D55E00",   # radial-associated state I — orange
    1: "#F0E442",   # radial-associated state II — yellow
    2: "#0072B2"    # mosaic-associated state — blue
}

# Expert-label colours are kept consistent with the
# conventional morphology classes.
expert_styles = {
    "Radial": {
        "colour": "#D55E00",
        "marker": "o"
    },
    "Mosaic": {
        "colour": "#0072B2"  ,
        "marker": "s"
    }
}


# ----------------------------------------------------------
# CREATE FIGURE
# ----------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.5, 6.3)
)


# ----------------------------------------------------------
# BACKGROUND GMM CLUSTERS
# ----------------------------------------------------------

for cluster_id in sorted(np.unique(labels)):

    cluster_mask = labels == cluster_id

    ax.scatter(
        emb_pca[cluster_mask, 0],
        emb_pca[cluster_mask, 1],
        s=8,
        color=cluster_colours[cluster_id],
        alpha=0.22,
        edgecolors="none",
        rasterized=True,
        zorder=1
    )


# ----------------------------------------------------------
# EXPERT-LABELLED PARTICLES
# ----------------------------------------------------------

for group_name in ["Radial", "Mosaic"]:

    indices = manual_group_indices.get(
        group_name,
        []
    )

    if len(indices) == 0:
        print(
            f"WARNING: no particles found for "
            f"expert group '{group_name}'.",
            flush=True
        )
        continue

    style = expert_styles[group_name]

   
    ax.scatter(
        emb_pca[indices, 0],
        emb_pca[indices, 1],
        s=55,
        marker=style["marker"],
        facecolors=style["colour"],
        edgecolors="white",
        linewidths=0.9,
        alpha=1.0,
        label=group_name,
        zorder=5
    )


# ----------------------------------------------------------
# AXIS LABELS
# ----------------------------------------------------------

ax.set_xlabel(
    f"PC1 ({var[0]:.1f}% explained variance)",
    labelpad=7
)

ax.set_ylabel(
    f"PC2 ({var[1]:.1f}% explained variance)",
    labelpad=7
)


# ----------------------------------------------------------
# AXIS APPEARANCE
# ----------------------------------------------------------

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.1)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=5,
    width=1.0,
    colors="black"
)

ax.grid(False)

ax.margins(
    x=0.025,
    y=0.025
)

ax.set_aspect(
    "equal",
    adjustable="box"
)


# ----------------------------------------------------------
# PANEL LABEL
# ----------------------------------------------------------



# ----------------------------------------------------------
# CUSTOM LEGEND
# ----------------------------------------------------------
# Show expert labels only. The coloured background already
# makes the GMM cluster structure visually apparent.

handles, legend_labels = ax.get_legend_handles_labels()

if handles:
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#AFAFAF",
        framealpha=0.95,
        fontsize=9,
        borderpad=0.45,
        labelspacing=0.35,
        handletextpad=0.4,
        handlelength=1.2,
        scatterpoints=1
    )

    legend.get_frame().set_linewidth(0.8)

# ----------------------------------------------------------
# REMOVE INTERNAL TITLE
# ----------------------------------------------------------

ax.set_title("")


# ----------------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------------

fig.tight_layout(
    pad=0.8
)

figure_base_path = os.path.join(
    NEW_OUTPUT_DIR,
    f"simclr_gmm_K{N_CLUSTERS}_pca_expert_labels_journal"
)

fig.savefig(
    figure_base_path + ".png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    figure_base_path + ".pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print(
    "Saved revised journal-ready PCA expert-label plot:",
    figure_base_path,
    flush=True
)
# ==========================================================
# SAVE 20 REPRESENTATIVE PARTICLES PER CLUSTER
# ==========================================================

REP_DIR = os.path.join(
    NEW_OUTPUT_DIR,
    "representative_particles"
)

os.makedirs(
    REP_DIR,
    exist_ok=True
)

for cl in sorted(np.unique(labels)):

    cluster_idx = np.where(
        labels == cl
    )[0]

    cluster_folder = os.path.join(
        REP_DIR,
        f"cluster_{cl}"
    )

    os.makedirs(
        cluster_folder,
        exist_ok=True
    )

    cluster_mean = gmm.means_[cl]
    cluster_variance = gmm.covariances_[cl]

    # Mahalanobis distance accounts for the variance of every
    # latent dimension within the assigned Gaussian component.
    differences = embeddings[cluster_idx] - cluster_mean

    distances = np.sqrt(
        np.sum(
            (differences ** 2) / cluster_variance,
            axis=1
        )
    )

    nearest_idx = cluster_idx[
        np.argsort(distances)[:50]
    ]

    print(
        f"Cluster {cl}: saving "
        f"{len(nearest_idx)} representatives",
        flush=True
    )

    for rank, idx in enumerate(nearest_idx):

        source_path = image_paths[idx]

        if not os.path.exists(source_path):

            print(
                f"WARNING: source image not found: "
                f"{source_path}",
                flush=True
            )

            continue

        save_name = (
            f"rank_{rank + 1}_"
            f"{os.path.basename(source_path)}"
        )

        save_path = os.path.join(
            cluster_folder,
            save_name
        )

        shutil.copy2(
            source_path,
            save_path
        )


# ==========================================================
# SAVE GMM POSTERIOR PROBABILITIES
# ==========================================================

posterior_df = pd.DataFrame({
    "image_path": image_paths,
    "image_name": image_names,
    "gmm_cluster": labels,
    "gmm_confidence": gmm_confidence
})

for c in range(N_CLUSTERS):

    posterior_df[
        f"posterior_cluster_{c}"
    ] = cluster_probabilities[:, c]

    posterior_df[
        f"posterior_cluster_{c}_percent"
    ] = cluster_probabilities[:, c] * 100

posterior_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"gmm_K{N_CLUSTERS}_posterior_probabilities.csv"
    ),
    index=False
)

posterior_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        f"gmm_K{N_CLUSTERS}_posterior_probabilities.xlsx"
    ),
    index=False
)

print(
    "Saved GMM posterior probabilities "
    "to Excel and CSV.",
    flush=True
)

print(
    f"Finished GMM K={N_CLUSTERS} clustering.",
    flush=True
)

print(
    "Results saved to:",
    NEW_OUTPUT_DIR,
    flush=True
)


# ==========================================================
# MODEL SELECTION: BEST NUMBER OF CLUSTERS
# ==========================================================

print("\n===============================================")
print("Evaluating number of clusters, K = 2 to 17")
print("===============================================\n")

model_selection_results = []

for k in range(2, 10):

    print(
        f"Evaluating K = {k}",
        flush=True
    )

    gmm_test = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42,
        n_init=10,
        max_iter=500,
        reg_covar=1e-6
    )

    labels_test = gmm_test.fit_predict(
        embeddings
    )

    unique_labels = np.unique(
        labels_test
    )

    bic = gmm_test.bic(
        embeddings
    )

    aic = gmm_test.aic(
        embeddings
    )

    if len(unique_labels) > 1:

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

    else:

        sil = np.nan
        db = np.nan
        ch = np.nan

    purity_scores = []

    for group_name, group_files in manual_groups.items():

        counts = np.zeros(
            k,
            dtype=int
        )

        total = 0

        for name in group_files:

            if name in name_to_index:

                idx = name_to_index[name]

                counts[
                    labels_test[idx]
                ] += 1

                total += 1

        if total > 0:

            purity_scores.append(
                counts.max() / total
            )

    if len(purity_scores) > 0:
        mean_purity = np.mean(
            purity_scores
        )
    else:
        mean_purity = np.nan

    model_selection_results.append({
        "K": k,
        "BIC": bic,
        "AIC": aic,
        "Silhouette": sil,
        "Davies_Bouldin": db,
        "Calinski_Harabasz": ch,
        "Manual_Purity": mean_purity
    })

model_selection_df = pd.DataFrame(
    model_selection_results
)

model_selection_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_model_selection_metrics.csv"
    ),
    index=False
)

print(
    model_selection_df,
    flush=True
)


# ==========================================================
# MODEL SELECTION PLOTS
# ==========================================================

metrics = [
    ("BIC", "Lower is better"),
    ("AIC", "Lower is better"),
    ("Silhouette", "Higher is better"),
    ("Davies_Bouldin", "Lower is better"),
    ("Calinski_Harabasz", "Higher is better"),
    ("Manual_Purity", "Higher is better")
]

for metric, subtitle in metrics:

    plt.figure(figsize=(6, 4))

    valid_df = model_selection_df.dropna(
        subset=[metric]
    )

    plt.plot(
        valid_df["K"],
        valid_df[metric],
        marker="o",
        linewidth=2
    )

    if metric in [
        "BIC",
        "AIC",
        "Davies_Bouldin"
    ]:

        best_k = valid_df.loc[
            valid_df[metric].idxmin(),
            "K"
        ]

    else:

        best_k = valid_df.loc[
            valid_df[metric].idxmax(),
            "K"
        ]

    plt.axvline(
        best_k,
        color="red",
        linestyle="--",
        label=f"Best K = {int(best_k)}"
    )

    plt.xlabel("Number of clusters, K")
    plt.ylabel(metric)

    plt.title(
        f"{metric}\n{subtitle}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            NEW_OUTPUT_DIR,
            f"gmm_{metric}_vs_K.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print(
    "\nModel selection finished.",
    flush=True
)

print("\n================================")
print(
    f"Number of particles analysed: "
    f"{embeddings.shape[0]}"
)
print(
    f"Embedding dimension: "
    f"{embeddings.shape[1]}"
)
print(
    f"Final number of GMM clusters: "
    f"{N_CLUSTERS}"
)
print("================================\n")

print(gmm.covariances_.shape)
print(gmm.covariances_)
print(np.min(gmm.covariances_))
print(np.max(gmm.covariances_))

# ==========================================================
# GMM POSTERIOR-PROBABILITY DIAGNOSTICS
# ==========================================================

print("\n===============================================")
print("GMM POSTERIOR-PROBABILITY DIAGNOSTICS")
print("===============================================\n")


# ----------------------------------------------------------
# 1. GMM COMPONENT WEIGHTS
# ----------------------------------------------------------

print("GMM component weights:")

for cl, weight in enumerate(gmm.weights_):
    print(
        f"Cluster {cl}: {weight:.8f} "
        f"({weight * 100:.3f}%)",
        flush=True
    )

print(
    "Sum of component weights:",
    gmm.weights_.sum(),
    flush=True
)


# ----------------------------------------------------------
# 2. MAXIMUM POSTERIOR-PROBABILITY SUMMARY
# ----------------------------------------------------------

max_posteriors = cluster_probabilities.max(axis=1)

posterior_percentiles = np.percentile(
    max_posteriors,
    [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
)

percentile_labels = [
    "Minimum",
    "1st percentile",
    "5th percentile",
    "10th percentile",
    "25th percentile",
    "Median",
    "75th percentile",
    "90th percentile",
    "95th percentile",
    "99th percentile",
    "Maximum"
]

print("\nMaximum posterior-probability distribution:")

for label, value in zip(
    percentile_labels,
    posterior_percentiles
):
    print(
        f"{label}: {value:.12f}",
        flush=True
    )


# ----------------------------------------------------------
# 3. COUNTS ABOVE CONFIDENCE THRESHOLDS
# ----------------------------------------------------------

confidence_thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999
]

confidence_summary = []

print("\nParticles above confidence thresholds:")

for threshold in confidence_thresholds:

    count = int(
        np.sum(max_posteriors >= threshold)
    )

    percentage = (
        100 * count / len(max_posteriors)
    )

    confidence_summary.append({
        "threshold": threshold,
        "n_particles": count,
        "percentage_particles": percentage
    })

    print(
        f">= {threshold:.5f}: "
        f"{count}/{len(max_posteriors)} "
        f"({percentage:.3f}%)",
        flush=True
    )

confidence_summary_df = pd.DataFrame(
    confidence_summary
)

confidence_summary_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_confidence_threshold_summary.csv"
    ),
    index=False
)


# ----------------------------------------------------------
# 4. CHECK NUMERICALLY EXTREME POSTERIORS
# ----------------------------------------------------------

n_exact_one = int(
    np.sum(max_posteriors == 1.0)
)

n_near_one = int(
    np.sum(max_posteriors >= 1 - 1e-12)
)

n_near_deterministic = int(
    np.sum(max_posteriors >= 0.999999)
)

print("\nNumerically extreme posterior assignments:")

print(
    f"Exactly equal to 1.0: "
    f"{n_exact_one}",
    flush=True
)

print(
    f"At least 1 - 1e-12: "
    f"{n_near_one}",
    flush=True
)

print(
    f"At least 0.999999: "
    f"{n_near_deterministic}",
    flush=True
)


# ----------------------------------------------------------
# 5. POSTERIOR ENTROPY
# ----------------------------------------------------------
# Entropy is low when one cluster dominates and high when
# probabilities are shared across several clusters.

probabilities_safe = np.clip(
    cluster_probabilities,
    1e-300,
    1.0
)

posterior_entropy = -np.sum(
    probabilities_safe
    * np.log(probabilities_safe),
    axis=1
)

# Normalize entropy to 0–1.
# 0 = deterministic assignment
# 1 = equal probabilities across all clusters

normalized_entropy = (
    posterior_entropy
    / np.log(N_CLUSTERS)
)

print("\nNormalized posterior entropy summary:")

entropy_percentiles = np.percentile(
    normalized_entropy,
    [0, 1, 5, 25, 50, 75, 95, 99, 100]
)

entropy_labels = [
    "Minimum",
    "1st percentile",
    "5th percentile",
    "25th percentile",
    "Median",
    "75th percentile",
    "95th percentile",
    "99th percentile",
    "Maximum"
]

for label, value in zip(
    entropy_labels,
    entropy_percentiles
):
    print(
        f"{label}: {value:.12f}",
        flush=True
    )


# ----------------------------------------------------------
# 6. PARTICLE LOG-LIKELIHOODS
# ----------------------------------------------------------

log_likelihood = gmm.score_samples(
    embeddings
)

print("\nPer-particle log-likelihood summary:")

log_likelihood_percentiles = np.percentile(
    log_likelihood,
    [0, 1, 5, 25, 50, 75, 95, 99, 100]
)

for label, value in zip(
    entropy_labels,
    log_likelihood_percentiles
):
    print(
        f"{label}: {value:.6f}",
        flush=True
    )


# ----------------------------------------------------------
# 7. MAHALANOBIS DISTANCE TO ASSIGNED COMPONENT
# ----------------------------------------------------------
# This is statistically consistent with covariance_type='diag'.

assigned_mahalanobis_sq = np.zeros(
    embeddings.shape[0],
    dtype=float
)

for i in range(embeddings.shape[0]):

    assigned_cluster = labels[i]

    cluster_mean = gmm.means_[
        assigned_cluster
    ]

    cluster_variance = gmm.covariances_[
        assigned_cluster
    ]

    assigned_mahalanobis_sq[i] = np.sum(
        (
            (
                embeddings[i]
                - cluster_mean
            ) ** 2
        )
        / cluster_variance
    )

assigned_mahalanobis = np.sqrt(
    assigned_mahalanobis_sq
)

print("\nAssigned-component Mahalanobis-distance summary:")

mahalanobis_percentiles = np.percentile(
    assigned_mahalanobis,
    [0, 1, 5, 25, 50, 75, 95, 99, 100]
)

for label, value in zip(
    entropy_labels,
    mahalanobis_percentiles
):
    print(
        f"{label}: {value:.6f}",
        flush=True
    )


# ----------------------------------------------------------
# 8. SECOND-HIGHEST POSTERIOR AND MARGIN
# ----------------------------------------------------------

sorted_probabilities = np.sort(
    cluster_probabilities,
    axis=1
)

highest_probability = (
    sorted_probabilities[:, -1]
)

second_highest_probability = (
    sorted_probabilities[:, -2]
)

posterior_margin = (
    highest_probability
    - second_highest_probability
)

print("\nPosterior-margin summary:")

margin_percentiles = np.percentile(
    posterior_margin,
    [0, 1, 5, 25, 50, 75, 95, 99, 100]
)

for label, value in zip(
    entropy_labels,
    margin_percentiles
):
    print(
        f"{label}: {value:.12f}",
        flush=True
    )


# ----------------------------------------------------------
# 9. SAVE COMPLETE DIAGNOSTIC TABLE
# ----------------------------------------------------------

diagnostic_df = pd.DataFrame({
    "image_path": image_paths,
    "image_name": image_names,
    "assigned_cluster": labels,
    "maximum_posterior": max_posteriors,
    "second_highest_posterior":
        second_highest_probability,
    "posterior_margin": posterior_margin,
    "posterior_entropy": posterior_entropy,
    "normalized_posterior_entropy":
        normalized_entropy,
    "log_likelihood": log_likelihood,
    "assigned_mahalanobis_distance":
        assigned_mahalanobis
})

for cl in range(N_CLUSTERS):

    diagnostic_df[
        f"posterior_cluster_{cl}"
    ] = cluster_probabilities[:, cl]

diagnostic_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_posterior_diagnostics.csv"
    ),
    index=False
)

diagnostic_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_posterior_diagnostics.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 10. SAVE MOST AMBIGUOUS PARTICLES
# ----------------------------------------------------------

N_AMBIGUOUS = min(
    100,
    len(diagnostic_df)
)

ambiguous_df = diagnostic_df.sort_values(
    by=[
        "maximum_posterior",
        "posterior_margin"
    ],
    ascending=[
        True,
        True
    ]
).head(N_AMBIGUOUS)

ambiguous_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_100_most_ambiguous_particles.csv"
    ),
    index=False
)

ambiguous_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_100_most_ambiguous_particles.xlsx"
    ),
    index=False
)

print(
    f"\nSaved the {N_AMBIGUOUS} most ambiguous "
    "particles.",
    flush=True
)


# ----------------------------------------------------------
# 11. SAVE MOST CENTRAL HIGH-CONFIDENCE PARTICLES
# ----------------------------------------------------------

central_summary = []

for cl in range(N_CLUSTERS):

    cluster_mask = labels == cl

    cluster_indices = np.where(
        cluster_mask
    )[0]

    cluster_table = diagnostic_df.iloc[
        cluster_indices
    ].copy()

    cluster_table = cluster_table.sort_values(
        by=[
            "assigned_mahalanobis_distance",
            "maximum_posterior"
        ],
        ascending=[
            True,
            False
        ]
    )

    cluster_table[
        "rank_within_cluster"
    ] = np.arange(
        1,
        len(cluster_table) + 1
    )

    central_summary.append(
        cluster_table.head(100)
    )

central_particles_df = pd.concat(
    central_summary,
    ignore_index=True
)

central_particles_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_central_particles_by_cluster.csv"
    ),
    index=False
)

central_particles_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_central_particles_by_cluster.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 12. MAXIMUM POSTERIOR HISTOGRAM
# ----------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(
    max_posteriors,
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
    f"GMM assignment confidence, K={N_CLUSTERS}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_maximum_posterior_histogram.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------
# 13. NORMALIZED ENTROPY HISTOGRAM
# ----------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(
    normalized_entropy,
    bins=50,
    edgecolor="black"
)

plt.xlabel(
    "Normalized posterior entropy"
)

plt.ylabel(
    "Number of particles"
)

plt.title(
    f"GMM posterior uncertainty, K={N_CLUSTERS}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_normalized_entropy_histogram.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------
# 14. POSTERIOR MARGIN HISTOGRAM
# ----------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.hist(
    posterior_margin,
    bins=50,
    edgecolor="black"
)

plt.xlabel(
    "Posterior probability margin"
)

plt.ylabel(
    "Number of particles"
)

plt.title(
    f"Difference between first and second "
    f"posterior probabilities, K={N_CLUSTERS}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_posterior_margin_histogram.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------
# 15. CONFIDENCE BY GMM COMPONENT
# ----------------------------------------------------------

confidence_by_cluster = []

for cl in range(N_CLUSTERS):

    cluster_values = max_posteriors[
        labels == cl
    ]

    confidence_by_cluster.append({
        "cluster": cl,
        "n_particles": len(cluster_values),
        "mean_confidence":
            np.mean(cluster_values),
        "median_confidence":
            np.median(cluster_values),
        "minimum_confidence":
            np.min(cluster_values),
        "percent_below_0_90":
            100 * np.mean(
                cluster_values < 0.90
            ),
        "percent_below_0_99":
            100 * np.mean(
                cluster_values < 0.99
            )
    })

confidence_by_cluster_df = pd.DataFrame(
    confidence_by_cluster
)

confidence_by_cluster_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_confidence_by_cluster.csv"
    ),
    index=False
)

print("\nConfidence by GMM component:")

print(
    confidence_by_cluster_df.to_string(
        index=False
    ),
    flush=True
)


# ----------------------------------------------------------
# 16. COMPARE COVARIANCE MODELS
# ----------------------------------------------------------
# This checks whether the extreme probabilities are specific
# to the diagonal covariance assumption.

covariance_models = [
    "diag",
    "tied",
    "full"
]

covariance_comparison = []

print("\n===============================================")
print("COMPARING GMM COVARIANCE MODELS")
print("===============================================\n")

for covariance_type in covariance_models:

    print(
        f"Testing covariance_type="
        f"{covariance_type}",
        flush=True
    )

    try:

        comparison_gmm = GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type=covariance_type,
            random_state=42,
            n_init=10,
            max_iter=500,
            reg_covar=1e-6
        )

        comparison_labels = (
            comparison_gmm.fit_predict(
                embeddings
            )
        )

        comparison_probabilities = (
            comparison_gmm.predict_proba(
                embeddings
            )
        )

        comparison_confidence = (
            comparison_probabilities.max(
                axis=1
            )
        )

        covariance_comparison.append({
            "covariance_type":
                covariance_type,
            "converged":
                comparison_gmm.converged_,
            "n_iterations":
                comparison_gmm.n_iter_,
            "BIC":
                comparison_gmm.bic(
                    embeddings
                ),
            "AIC":
                comparison_gmm.aic(
                    embeddings
                ),
            "mean_max_posterior":
                np.mean(
                    comparison_confidence
                ),
            "median_max_posterior":
                np.median(
                    comparison_confidence
                ),
            "minimum_max_posterior":
                np.min(
                    comparison_confidence
                ),
            "percent_at_least_0_99":
                100 * np.mean(
                    comparison_confidence
                    >= 0.99
                ),
            "percent_at_least_0_999999":
                100 * np.mean(
                    comparison_confidence
                    >= 0.999999
                ),
            "mean_normalized_entropy":
                np.mean(
                    -np.sum(
                        np.clip(
                            comparison_probabilities,
                            1e-300,
                            1.0
                        )
                        * np.log(
                            np.clip(
                                comparison_probabilities,
                                1e-300,
                                1.0
                            )
                        ),
                        axis=1
                    )
                    / np.log(N_CLUSTERS)
                )
        })

    except Exception as error:

        print(
            f"WARNING: covariance model "
            f"{covariance_type} failed:",
            error,
            flush=True
        )

covariance_comparison_df = pd.DataFrame(
    covariance_comparison
)

covariance_comparison_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_covariance_model_comparison.csv"
    ),
    index=False
)

covariance_comparison_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "gmm_covariance_model_comparison.xlsx"
    ),
    index=False
)

print("\nCovariance-model comparison:")

print(
    covariance_comparison_df.to_string(
        index=False
    ),
    flush=True
)


# ----------------------------------------------------------
# 17. FINAL DIAGNOSTIC MESSAGE
# ----------------------------------------------------------

print("\n===============================================")
print("POSTERIOR DIAGNOSTICS FINISHED")
print("===============================================")

print(
    "Diagnostic files saved to:",
    NEW_OUTPUT_DIR,
    flush=True
)

# ==========================================================
# VALIDATION DATASET AUDIT
# ==========================================================

from collections import Counter

print("\n===============================================")
print("VALIDATION DATASET AUDIT")
print("===============================================\n")


# ----------------------------------------------------------
# 1. BUILD RADIAL AND MOSAIC REFERENCE LISTS
# ----------------------------------------------------------

radial_files_raw = (
    GROUP_1
    + GROUP_2
    + GROUP_3
    + GROUP_5
)

mosaic_files_raw = (
    GROUP_4
    + GROUP_6
)


# ----------------------------------------------------------
# 2. CHECK REQUESTED COUNTS
# ----------------------------------------------------------

print("REQUESTED VALIDATION PARTICLES")
print("--------------------------------")

print(
    f"Radial entries requested: "
    f"{len(radial_files_raw)}"
)

print(
    f"Mosaic entries requested: "
    f"{len(mosaic_files_raw)}"
)

print(
    f"Total entries requested: "
    f"{len(radial_files_raw) + len(mosaic_files_raw)}"
)


# ----------------------------------------------------------
# 3. IDENTIFY DUPLICATES WITHIN EACH CLASS
# ----------------------------------------------------------

radial_counter = Counter(
    radial_files_raw
)

mosaic_counter = Counter(
    mosaic_files_raw
)

radial_duplicates = {
    name: count
    for name, count in radial_counter.items()
    if count > 1
}

mosaic_duplicates = {
    name: count
    for name, count in mosaic_counter.items()
    if count > 1
}


print("\nDUPLICATE FILENAMES")
print("--------------------------------")

print(
    f"Duplicated radial filenames: "
    f"{len(radial_duplicates)}"
)

for name, count in radial_duplicates.items():

    print(
        f"  {name} -> appears {count} times"
    )


print(
    f"\nDuplicated mosaic filenames: "
    f"{len(mosaic_duplicates)}"
)

for name, count in mosaic_duplicates.items():

    print(
        f"  {name} -> appears {count} times"
    )


# ----------------------------------------------------------
# 4. REMOVE DUPLICATES WHILE PRESERVING ORDER
# ----------------------------------------------------------

radial_files_unique = list(
    dict.fromkeys(
        radial_files_raw
    )
)

mosaic_files_unique = list(
    dict.fromkeys(
        mosaic_files_raw
    )
)


print("\nUNIQUE REQUESTED PARTICLES")
print("--------------------------------")

print(
    f"Unique radial filenames: "
    f"{len(radial_files_unique)}"
)

print(
    f"Unique mosaic filenames: "
    f"{len(mosaic_files_unique)}"
)

print(
    f"Unique total filenames: "
    f"{len(radial_files_unique) + len(mosaic_files_unique)}"
)


# ----------------------------------------------------------
# 5. CHECK FOR PARTICLES LISTED IN BOTH CLASSES
# ----------------------------------------------------------

radial_set = set(
    radial_files_unique
)

mosaic_set = set(
    mosaic_files_unique
)

cross_class_overlap = sorted(
    radial_set.intersection(
        mosaic_set
    )
)


print("\nCROSS-CLASS OVERLAP")
print("--------------------------------")

print(
    f"Particles appearing in both radial "
    f"and mosaic lists: "
    f"{len(cross_class_overlap)}"
)

for name in cross_class_overlap:

    print(
        f"  {name}"
    )


# ----------------------------------------------------------
# 6. IDENTIFY PARTICLES MISSING FROM EMBEDDING DATASET
# ----------------------------------------------------------

available_names = set(
    image_names
)

missing_radial = [
    name
    for name in radial_files_unique
    if name not in available_names
]

missing_mosaic = [
    name
    for name in mosaic_files_unique
    if name not in available_names
]


print("\nMISSING PARTICLES")
print("--------------------------------")

print(
    f"Missing radial particles: "
    f"{len(missing_radial)}"
)

for name in missing_radial:

    print(
        f"  {name}"
    )


print(
    f"\nMissing mosaic particles: "
    f"{len(missing_mosaic)}"
)

for name in missing_mosaic:

    print(
        f"  {name}"
    )


# ----------------------------------------------------------
# 7. CREATE FINAL USABLE VALIDATION LISTS
# ----------------------------------------------------------

usable_radial = [
    name
    for name in radial_files_unique
    if (
        name in available_names
        and name not in cross_class_overlap
    )
]

usable_mosaic = [
    name
    for name in mosaic_files_unique
    if (
        name in available_names
        and name not in cross_class_overlap
    )
]


print("\nFINAL USABLE VALIDATION DATASET")
print("--------------------------------")

print(
    f"Usable radial particles: "
    f"{len(usable_radial)}"
)

print(
    f"Usable mosaic particles: "
    f"{len(usable_mosaic)}"
)

print(
    f"Total usable particles: "
    f"{len(usable_radial) + len(usable_mosaic)}"
)


# ----------------------------------------------------------
# 8. CREATE PARTICLE-LEVEL AUDIT TABLE
# ----------------------------------------------------------

audit_rows = []

for name in radial_files_unique:

    audit_rows.append({
        "image_name": name,
        "manual_class": "Radial",
        "found_in_embedding_dataset":
            name in available_names,
        "duplicated_within_class":
            name in radial_duplicates,
        "duplicate_count":
            radial_counter[name],
        "appears_in_both_classes":
            name in cross_class_overlap,
        "usable_for_validation":
            (
                name in available_names
                and name not in cross_class_overlap
            )
    })


for name in mosaic_files_unique:

    audit_rows.append({
        "image_name": name,
        "manual_class": "Mosaic",
        "found_in_embedding_dataset":
            name in available_names,
        "duplicated_within_class":
            name in mosaic_duplicates,
        "duplicate_count":
            mosaic_counter[name],
        "appears_in_both_classes":
            name in cross_class_overlap,
        "usable_for_validation":
            (
                name in available_names
                and name not in cross_class_overlap
            )
    })


validation_audit_df = pd.DataFrame(
    audit_rows
)


validation_audit_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "validation_dataset_audit.csv"
    ),
    index=False
)


validation_audit_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "validation_dataset_audit.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 9. SAVE MISSING PARTICLES SEPARATELY
# ----------------------------------------------------------

missing_rows = []

for name in missing_radial:

    missing_rows.append({
        "image_name": name,
        "manual_class": "Radial"
    })


for name in missing_mosaic:

    missing_rows.append({
        "image_name": name,
        "manual_class": "Mosaic"
    })


missing_particles_df = pd.DataFrame(
    missing_rows
)


missing_particles_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "missing_validation_particles.csv"
    ),
    index=False
)


missing_particles_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "missing_validation_particles.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 10. SAVE DUPLICATES SEPARATELY
# ----------------------------------------------------------

duplicate_rows = []

for name, count in radial_duplicates.items():

    duplicate_rows.append({
        "image_name": name,
        "manual_class": "Radial",
        "number_of_occurrences": count
    })


for name, count in mosaic_duplicates.items():

    duplicate_rows.append({
        "image_name": name,
        "manual_class": "Mosaic",
        "number_of_occurrences": count
    })


duplicates_df = pd.DataFrame(
    duplicate_rows
)


duplicates_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "duplicate_validation_particles.csv"
    ),
    index=False
)


duplicates_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "duplicate_validation_particles.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 11. SAVE SUMMARY TABLE
# ----------------------------------------------------------

validation_summary_df = pd.DataFrame({
    "Metric": [
        "Requested radial entries",
        "Requested mosaic entries",
        "Requested total entries",
        "Unique radial filenames",
        "Unique mosaic filenames",
        "Unique total filenames",
        "Duplicated radial filenames",
        "Duplicated mosaic filenames",
        "Cross-class overlap",
        "Missing radial particles",
        "Missing mosaic particles",
        "Usable radial particles",
        "Usable mosaic particles",
        "Total usable particles"
    ],
    "Value": [
        len(radial_files_raw),
        len(mosaic_files_raw),
        (
            len(radial_files_raw)
            + len(mosaic_files_raw)
        ),
        len(radial_files_unique),
        len(mosaic_files_unique),
        (
            len(radial_files_unique)
            + len(mosaic_files_unique)
        ),
        len(radial_duplicates),
        len(mosaic_duplicates),
        len(cross_class_overlap),
        len(missing_radial),
        len(missing_mosaic),
        len(usable_radial),
        len(usable_mosaic),
        (
            len(usable_radial)
            + len(usable_mosaic)
        )
    ]
})


validation_summary_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "validation_dataset_summary.csv"
    ),
    index=False
)


validation_summary_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "validation_dataset_summary.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 12. FINAL CONSOLE SUMMARY
# ----------------------------------------------------------

print("\n===============================================")
print("VALIDATION DATASET FINAL SUMMARY")
print("===============================================")

print(
    f"Requested total entries: "
    f"{len(radial_files_raw) + len(mosaic_files_raw)}"
)

print(
    f"Unique requested particles: "
    f"{len(radial_files_unique) + len(mosaic_files_unique)}"
)

print(
    f"Missing particles: "
    f"{len(missing_radial) + len(missing_mosaic)}"
)

print(
    f"Cross-class overlaps removed: "
    f"{len(cross_class_overlap)}"
)

print(
    f"Final radial particles: "
    f"{len(usable_radial)}"
)

print(
    f"Final mosaic particles: "
    f"{len(usable_mosaic)}"
)

print(
    f"Final validation sample size: "
    f"{len(usable_radial) + len(usable_mosaic)}"
)

print(
    "\nValidation audit files saved to:",
    NEW_OUTPUT_DIR,
    flush=True
)

# ==========================================================
# CLASSIFICATION PERFORMANCE (ONLY FOR K = 2)
# ==========================================================

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report
)

if N_CLUSTERS == 2:

    print("\n===========================================")
    print("RADIAL–MOSAIC CLASSIFICATION PERFORMANCE")
    print("===========================================\n")

    # ------------------------------------------------------
    # Create ground-truth labels
    # Radial = 0
    # Mosaic = 1
    # ------------------------------------------------------

    y_true = []
    y_pred = []

    radial_files = set(usable_radial)
    mosaic_files = set(usable_mosaic)
    
    for name in image_names:

        if name in radial_files:

            y_true.append(0)

            idx = name_to_index[name]
            y_pred.append(labels[idx])

        elif name in mosaic_files:

            y_true.append(1)

            idx = name_to_index[name]
            y_pred.append(labels[idx])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nCLASSIFICATION SAMPLE SIZE")
    print("--------------------------------")
    print("Total evaluated:", len(y_true))
    print("True radial:", np.sum(y_true == 0))
    print("True mosaic:", np.sum(y_true == 1))

    # ------------------------------------------------------
    # Make sure cluster numbering matches manual labels
    # ------------------------------------------------------

    cm = confusion_matrix(y_true, y_pred)

    # If clusters are reversed, flip them
    if cm.trace() < np.fliplr(cm).trace():
        y_pred = 1 - y_pred
        cm = confusion_matrix(y_true, y_pred)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    print("Confusion Matrix")
    print(cm)

    print("\nPerformance metrics")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Balanced Accuracy  : {balanced_acc:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1-score           : {f1:.4f}")
    print(f"Cohen's Kappa      : {kappa:.4f}")

    print("\nDetailed Classification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Radial", "Mosaic"],
            digits=4
        )
    )

    # ------------------------------------------------------
    # Save confusion matrix
    # ------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=["True Radial", "True Mosaic"],
        columns=["Predicted Radial", "Predicted Mosaic"]
    )

    cm_df.to_csv(
        os.path.join(
            NEW_OUTPUT_DIR,
            "confusion_matrix.csv"
        )
    )

    # ------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Cohen's Kappa"
        ],
        "Value": [
            accuracy,
            balanced_acc,
            precision,
            recall,
            f1,
            kappa
        ]
    })

    metrics_df.to_csv(
        os.path.join(
            NEW_OUTPUT_DIR,
            "classification_metrics.csv"
        ),
        index=False
    )

    metrics_df.to_excel(
        os.path.join(
            NEW_OUTPUT_DIR,
            "classification_metrics.xlsx"
        ),
        index=False
    )

    print("\nClassification metrics saved.")


# ==========================================================
# STANDARDIZED METRICS FOR SIMCLR–AUTOENCODER COMPARISON
# ==========================================================
# Run the same block, with the same GMM settings, for the
# 512-D autoencoder embeddings. The resulting one-row CSV is
# designed to be combined directly across the two methods.
#
# Main comparison metrics:
#   1. Internal cluster structure:
#      silhouette, Davies–Bouldin, Calinski–Harabasz
#   2. GMM fit:
#      BIC, AIC, mean/median per-particle log-likelihood
#   3. Assignment certainty:
#      maximum posterior, posterior entropy and posterior margin
#   4. Cluster balance:
#      cluster-size entropy and smallest/largest cluster ratio
#   5. GMM stability:
#      ARI and NMI across repeated random initialisations
#   6. Representation compactness:
#      PCA variance concentration and PCs needed for 80/90/95%
#   7. Generalisation:
#      train–unseen likelihood, confidence, entropy, margin and
#      cluster-proportion differences.
#
# IMPORTANT:
# BIC, AIC and log-likelihood are directly comparable only
# because both methods use:
#   - the same particles,
#   - the same retained dimensionality (512),
#   - the same K,
#   - and the same GMM covariance specification.

print("\n===============================================")
print("STANDARDIZED SIMCLR–AUTOENCODER METRICS")
print("===============================================\n")

MODEL_NAME = "SimCLR_512"
STABILITY_RUNS = 10
STABILITY_N_INIT = 5
STABILITY_RANDOM_SEEDS = list(range(STABILITY_RUNS))


def safe_sample_std(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    return float(np.std(values, ddof=1))


def standardized_mean_difference(group_a, group_b):
    """
    Absolute standardized mean difference using pooled sample SD.
    Smaller values indicate closer train/unseen distributions.
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)

    pooled_variance = (
        ((len(group_a) - 1) * var_a)
        + ((len(group_b) - 1) * var_b)
    ) / (
        len(group_a) + len(group_b) - 2
    )

    if pooled_variance <= 0:
        return np.nan

    return float(
        abs(np.mean(group_a) - np.mean(group_b))
        / np.sqrt(pooled_variance)
    )


def normalized_cluster_entropy(cluster_labels, n_clusters):
    """
    Returns a value from 0 to 1.
    1 means equally populated clusters.
    0 means all particles occupy one cluster.
    """
    counts = np.bincount(
        cluster_labels,
        minlength=n_clusters
    ).astype(float)

    proportions = counts / counts.sum()
    positive = proportions[proportions > 0]

    entropy = -np.sum(
        positive * np.log(positive)
    )

    if n_clusters <= 1:
        return np.nan

    return float(
        entropy / np.log(n_clusters)
    )


def posterior_diagnostics(probability_matrix):
    safe_probabilities = np.clip(
        probability_matrix,
        1e-300,
        1.0
    )

    maximum = probability_matrix.max(axis=1)

    sorted_probabilities = np.sort(
        probability_matrix,
        axis=1
    )

    second_highest = sorted_probabilities[:, -2]

    margin = maximum - second_highest

    entropy = -np.sum(
        safe_probabilities
        * np.log(safe_probabilities),
        axis=1
    )

    normalized = (
        entropy
        / np.log(probability_matrix.shape[1])
    )

    return {
        "maximum": maximum,
        "second_highest": second_highest,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized
    }


# ----------------------------------------------------------
# 1. INTERNAL CLUSTER-QUALITY METRICS
# ----------------------------------------------------------

comparison_silhouette = silhouette_score(
    embeddings,
    labels
)

comparison_davies_bouldin = davies_bouldin_score(
    embeddings,
    labels
)

comparison_calinski_harabasz = calinski_harabasz_score(
    embeddings,
    labels
)


# ----------------------------------------------------------
# 2. GMM FIT METRICS
# ----------------------------------------------------------

comparison_bic = gmm.bic(
    embeddings
)

comparison_aic = gmm.aic(
    embeddings
)

comparison_training_log_likelihood = gmm.score_samples(
    embeddings
)

comparison_mean_log_likelihood = np.mean(
    comparison_training_log_likelihood
)

comparison_median_log_likelihood = np.median(
    comparison_training_log_likelihood
)

comparison_std_log_likelihood = safe_sample_std(
    comparison_training_log_likelihood
)


# ----------------------------------------------------------
# 3. POSTERIOR-CONFIDENCE METRICS
# ----------------------------------------------------------

training_posterior_metrics = posterior_diagnostics(
    cluster_probabilities
)

comparison_max_posterior = training_posterior_metrics[
    "maximum"
]

comparison_posterior_margin = training_posterior_metrics[
    "margin"
]

comparison_normalized_entropy = training_posterior_metrics[
    "normalized_entropy"
]


# ----------------------------------------------------------
# 4. CLUSTER-BALANCE METRICS
# ----------------------------------------------------------

comparison_cluster_counts = np.bincount(
    labels,
    minlength=N_CLUSTERS
)

comparison_cluster_proportions = (
    comparison_cluster_counts
    / comparison_cluster_counts.sum()
)

comparison_cluster_entropy = normalized_cluster_entropy(
    labels,
    N_CLUSTERS
)

comparison_min_max_cluster_ratio = (
    comparison_cluster_counts.min()
    / comparison_cluster_counts.max()
)


# ----------------------------------------------------------
# 5. ASSIGNED-COMPONENT MAHALANOBIS DISTANCE
# ----------------------------------------------------------

comparison_mahalanobis_squared = np.zeros(
    embeddings.shape[0],
    dtype=float
)

for row_index in range(embeddings.shape[0]):
    assigned_component = labels[row_index]

    difference = (
        embeddings[row_index]
        - gmm.means_[assigned_component]
    )

    variance = gmm.covariances_[
        assigned_component
    ]

    comparison_mahalanobis_squared[
        row_index
    ] = np.sum(
        (difference ** 2)
        / variance
    )

comparison_mahalanobis = np.sqrt(
    comparison_mahalanobis_squared
)


# ----------------------------------------------------------
# 6. REPRESENTATION COMPACTNESS USING PCA
# ----------------------------------------------------------
# These are descriptive supporting metrics. They should not
# be interpreted alone as proof that one representation is
# superior.

comparison_pca = PCA(
    n_components=min(
        embeddings.shape[0],
        embeddings.shape[1]
    ),
    svd_solver="full"
)

comparison_pca.fit(
    embeddings
)

comparison_cumulative_variance = np.cumsum(
    comparison_pca.explained_variance_ratio_
)


def components_needed(cumulative_variance, threshold):
    return int(
        np.searchsorted(
            cumulative_variance,
            threshold
        )
        + 1
    )


comparison_pc1_variance = (
    comparison_pca.explained_variance_ratio_[0]
)

comparison_pc2_variance = (
    comparison_pca.explained_variance_ratio_[1]
    if len(comparison_pca.explained_variance_ratio_) >= 2
    else np.nan
)

comparison_first2_variance = (
    comparison_cumulative_variance[1]
    if len(comparison_cumulative_variance) >= 2
    else comparison_cumulative_variance[0]
)

comparison_first5_variance = (
    comparison_cumulative_variance[
        min(4, len(comparison_cumulative_variance) - 1)
    ]
)

comparison_pcs_80 = components_needed(
    comparison_cumulative_variance,
    0.80
)

comparison_pcs_90 = components_needed(
    comparison_cumulative_variance,
    0.90
)

comparison_pcs_95 = components_needed(
    comparison_cumulative_variance,
    0.95
)


# ----------------------------------------------------------
# 7. GMM STABILITY ACROSS RANDOM INITIALISATIONS
# ----------------------------------------------------------
# The fitted labels are compared with the final reference GMM.
# High ARI and NMI indicate that the clustering solution is
# not strongly dependent on initialisation.

stability_rows = []

for stability_seed in STABILITY_RANDOM_SEEDS:
    stability_gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="diag",
        random_state=stability_seed,
        n_init=STABILITY_N_INIT,
        max_iter=500,
        reg_covar=1e-6
    )

    stability_labels = stability_gmm.fit_predict(
        embeddings
    )

    stability_probabilities = stability_gmm.predict_proba(
        embeddings
    )

    stability_rows.append({
        "model": MODEL_NAME,
        "random_seed": stability_seed,
        "adjusted_rand_index_vs_reference":
            adjusted_rand_score(
                labels,
                stability_labels
            ),
        "normalized_mutual_information_vs_reference":
            normalized_mutual_info_score(
                labels,
                stability_labels
            ),
        "BIC": stability_gmm.bic(
            embeddings
        ),
        "AIC": stability_gmm.aic(
            embeddings
        ),
        "mean_log_likelihood":
            stability_gmm.score(
                embeddings
            ),
        "mean_maximum_posterior":
            np.mean(
                stability_probabilities.max(
                    axis=1
                )
            ),
        "converged": stability_gmm.converged_,
        "n_iterations": stability_gmm.n_iter_
    })

stability_df = pd.DataFrame(
    stability_rows
)

stability_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "comparison_gmm_stability_runs.csv"
    ),
    index=False
)

stability_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "comparison_gmm_stability_runs.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 8. SAVE ONE-ROW TRAINING COMPARISON TABLE
# ----------------------------------------------------------

comparison_metrics_row = {
    "model": MODEL_NAME,
    "n_particles": embeddings.shape[0],
    "embedding_dimension": embeddings.shape[1],
    "n_clusters": N_CLUSTERS,
    "covariance_type": gmm.covariance_type,

    "silhouette_score":
        comparison_silhouette,
    "davies_bouldin_score":
        comparison_davies_bouldin,
    "calinski_harabasz_score":
        comparison_calinski_harabasz,

    "BIC": comparison_bic,
    "BIC_per_particle":
        comparison_bic / embeddings.shape[0],
    "AIC": comparison_aic,
    "AIC_per_particle":
        comparison_aic / embeddings.shape[0],
    "mean_log_likelihood":
        comparison_mean_log_likelihood,
    "median_log_likelihood":
        comparison_median_log_likelihood,
    "std_log_likelihood":
        comparison_std_log_likelihood,

    "mean_maximum_posterior":
        np.mean(comparison_max_posterior),
    "median_maximum_posterior":
        np.median(comparison_max_posterior),
    "minimum_maximum_posterior":
        np.min(comparison_max_posterior),
    "percent_maximum_posterior_at_least_0_95":
        100 * np.mean(
            comparison_max_posterior >= 0.95
        ),
    "percent_maximum_posterior_at_least_0_99":
        100 * np.mean(
            comparison_max_posterior >= 0.99
        ),
    "mean_normalized_posterior_entropy":
        np.mean(
            comparison_normalized_entropy
        ),
    "median_normalized_posterior_entropy":
        np.median(
            comparison_normalized_entropy
        ),
    "mean_posterior_margin":
        np.mean(
            comparison_posterior_margin
        ),
    "median_posterior_margin":
        np.median(
            comparison_posterior_margin
        ),

    "normalized_cluster_size_entropy":
        comparison_cluster_entropy,
    "smallest_to_largest_cluster_size_ratio":
        comparison_min_max_cluster_ratio,

    "mean_assigned_mahalanobis_distance":
        np.mean(comparison_mahalanobis),
    "median_assigned_mahalanobis_distance":
        np.median(comparison_mahalanobis),

    "PC1_variance_fraction":
        comparison_pc1_variance,
    "PC2_variance_fraction":
        comparison_pc2_variance,
    "first_2_PCs_variance_fraction":
        comparison_first2_variance,
    "first_5_PCs_variance_fraction":
        comparison_first5_variance,
    "PCs_needed_for_80_percent_variance":
        comparison_pcs_80,
    "PCs_needed_for_90_percent_variance":
        comparison_pcs_90,
    "PCs_needed_for_95_percent_variance":
        comparison_pcs_95,

    "mean_stability_ARI":
        stability_df[
            "adjusted_rand_index_vs_reference"
        ].mean(),
    "minimum_stability_ARI":
        stability_df[
            "adjusted_rand_index_vs_reference"
        ].min(),
    "mean_stability_NMI":
        stability_df[
            "normalized_mutual_information_vs_reference"
        ].mean(),
    "minimum_stability_NMI":
        stability_df[
            "normalized_mutual_information_vs_reference"
        ].min(),
    "fraction_stability_runs_converged":
        stability_df["converged"].mean()
}

for component_index, component_count in enumerate(
    comparison_cluster_counts
):
    comparison_metrics_row[
        f"cluster_{component_index}_n"
    ] = int(component_count)

    comparison_metrics_row[
        f"cluster_{component_index}_fraction"
    ] = comparison_cluster_proportions[
        component_index
    ]

comparison_metrics_df = pd.DataFrame([
    comparison_metrics_row
])

comparison_metrics_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "simclr_512_model_comparison_metrics.csv"
    ),
    index=False
)

comparison_metrics_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "simclr_512_model_comparison_metrics.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 9. TRAIN–UNSEEN GENERALISATION METRICS
# ----------------------------------------------------------

unseen_posterior_metrics = posterior_diagnostics(
    unseen_probabilities
)

unseen_max_posterior = unseen_posterior_metrics[
    "maximum"
]

unseen_posterior_margin = unseen_posterior_metrics[
    "margin"
]

unseen_normalized_entropy = unseen_posterior_metrics[
    "normalized_entropy"
]

training_cluster_proportions = (
    np.bincount(
        labels,
        minlength=N_CLUSTERS
    ).astype(float)
)

training_cluster_proportions /= (
    training_cluster_proportions.sum()
)

unseen_cluster_proportions = (
    np.bincount(
        unseen_labels,
        minlength=N_CLUSTERS
    ).astype(float)
)

unseen_cluster_proportions /= (
    unseen_cluster_proportions.sum()
)

# Jensen–Shannon distance is finite and symmetric.
# 0 means identical cluster proportions.
cluster_proportion_js_distance = jensenshannon(
    training_cluster_proportions,
    unseen_cluster_proportions,
    base=2
)

train_unseen_ks = ks_2samp(
    training_log_likelihood,
    unseen_log_likelihood,
    alternative="two-sided",
    mode="auto"
)

train_unseen_wasserstein = wasserstein_distance(
    training_log_likelihood,
    unseen_log_likelihood
)

training_ll_sd = safe_sample_std(
    training_log_likelihood
)

normalized_wasserstein = (
    train_unseen_wasserstein
    / training_ll_sd
    if training_ll_sd > 0
    else np.nan
)

generalisation_metrics_row = {
    "model": MODEL_NAME,
    "n_training_particles":
        len(training_log_likelihood),
    "n_unseen_particles":
        len(unseen_log_likelihood),

    "training_mean_log_likelihood":
        np.mean(training_log_likelihood),
    "unseen_mean_log_likelihood":
        np.mean(unseen_log_likelihood),
    "absolute_mean_log_likelihood_gap":
        abs(
            np.mean(training_log_likelihood)
            - np.mean(unseen_log_likelihood)
        ),
    "training_median_log_likelihood":
        np.median(training_log_likelihood),
    "unseen_median_log_likelihood":
        np.median(unseen_log_likelihood),
    "absolute_median_log_likelihood_gap":
        abs(
            np.median(training_log_likelihood)
            - np.median(unseen_log_likelihood)
        ),
    "log_likelihood_standardized_mean_difference":
        standardized_mean_difference(
            training_log_likelihood,
            unseen_log_likelihood
        ),
    "log_likelihood_KS_statistic":
        train_unseen_ks.statistic,
    "log_likelihood_KS_p_value":
        train_unseen_ks.pvalue,
    "log_likelihood_wasserstein_distance":
        train_unseen_wasserstein,
    "log_likelihood_normalized_wasserstein_distance":
        normalized_wasserstein,

    "training_mean_maximum_posterior":
        np.mean(
            comparison_max_posterior
        ),
    "unseen_mean_maximum_posterior":
        np.mean(
            unseen_max_posterior
        ),
    "absolute_mean_maximum_posterior_gap":
        abs(
            np.mean(comparison_max_posterior)
            - np.mean(unseen_max_posterior)
        ),
    "training_median_maximum_posterior":
        np.median(
            comparison_max_posterior
        ),
    "unseen_median_maximum_posterior":
        np.median(
            unseen_max_posterior
        ),
    "training_percent_posterior_at_least_0_95":
        100 * np.mean(
            comparison_max_posterior >= 0.95
        ),
    "unseen_percent_posterior_at_least_0_95":
        100 * np.mean(
            unseen_max_posterior >= 0.95
        ),
    "training_percent_posterior_at_least_0_99":
        100 * np.mean(
            comparison_max_posterior >= 0.99
        ),
    "unseen_percent_posterior_at_least_0_99":
        100 * np.mean(
            unseen_max_posterior >= 0.99
        ),

    "training_mean_normalized_entropy":
        np.mean(
            comparison_normalized_entropy
        ),
    "unseen_mean_normalized_entropy":
        np.mean(
            unseen_normalized_entropy
        ),
    "absolute_mean_normalized_entropy_gap":
        abs(
            np.mean(comparison_normalized_entropy)
            - np.mean(unseen_normalized_entropy)
        ),

    "training_mean_posterior_margin":
        np.mean(
            comparison_posterior_margin
        ),
    "unseen_mean_posterior_margin":
        np.mean(
            unseen_posterior_margin
        ),
    "absolute_mean_posterior_margin_gap":
        abs(
            np.mean(comparison_posterior_margin)
            - np.mean(unseen_posterior_margin)
        ),

    "unseen_percent_below_training_1_percent_LL_threshold":
        100 * np.mean(
            unseen_log_likelihood
            < np.quantile(
                training_log_likelihood,
                0.01
            )
        ),
    "unseen_percent_below_training_5_percent_LL_threshold":
        100 * np.mean(
            unseen_log_likelihood
            < np.quantile(
                training_log_likelihood,
                0.05
            )
        ),

    "cluster_proportion_Jensen_Shannon_distance":
        cluster_proportion_js_distance,
    "training_normalized_cluster_size_entropy":
        normalized_cluster_entropy(
            labels,
            N_CLUSTERS
        ),
    "unseen_normalized_cluster_size_entropy":
        normalized_cluster_entropy(
            unseen_labels,
            N_CLUSTERS
        )
}

for component_index in range(N_CLUSTERS):
    generalisation_metrics_row[
        f"training_cluster_{component_index}_fraction"
    ] = training_cluster_proportions[
        component_index
    ]

    generalisation_metrics_row[
        f"unseen_cluster_{component_index}_fraction"
    ] = unseen_cluster_proportions[
        component_index
    ]

generalisation_metrics_df = pd.DataFrame([
    generalisation_metrics_row
])

generalisation_metrics_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "simclr_512_generalisation_comparison_metrics.csv"
    ),
    index=False
)

generalisation_metrics_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "simclr_512_generalisation_comparison_metrics.xlsx"
    ),
    index=False
)


# ----------------------------------------------------------
# 10. HUMAN-READABLE REDUCED TABLE FOR THE PAPER
# ----------------------------------------------------------
# This table contains the most interpretable metrics. The full
# CSVs above preserve all diagnostics for supplementary use.

paper_metrics_df = pd.DataFrame([
    {
        "Metric": "Silhouette score",
        "Value": comparison_silhouette,
        "Preferred direction": "Higher"
    },
    {
        "Metric": "Davies–Bouldin index",
        "Value": comparison_davies_bouldin,
        "Preferred direction": "Lower"
    },
    {
        "Metric": "Calinski–Harabasz index",
        "Value": comparison_calinski_harabasz,
        "Preferred direction": "Higher"
    },
    {
        "Metric": "BIC per particle",
        "Value": comparison_bic / embeddings.shape[0],
        "Preferred direction": "Lower"
    },
    {
        "Metric": "AIC per particle",
        "Value": comparison_aic / embeddings.shape[0],
        "Preferred direction": "Lower"
    },
    {
        "Metric": "Mean log-likelihood",
        "Value": comparison_mean_log_likelihood,
        "Preferred direction": "Higher"
    },
    {
        "Metric": "Mean maximum posterior probability",
        "Value": np.mean(comparison_max_posterior),
        "Preferred direction": "Higher"
    },
    {
        "Metric": "Mean normalized posterior entropy",
        "Value": np.mean(comparison_normalized_entropy),
        "Preferred direction": "Lower"
    },
    {
        "Metric": "Mean posterior margin",
        "Value": np.mean(comparison_posterior_margin),
        "Preferred direction": "Higher"
    },
    {
        "Metric": "Mean GMM stability ARI",
        "Value": stability_df[
            "adjusted_rand_index_vs_reference"
        ].mean(),
        "Preferred direction": "Higher"
    },
    {
        "Metric": "Train–unseen LL standardized difference",
        "Value": generalisation_metrics_row[
            "log_likelihood_standardized_mean_difference"
        ],
        "Preferred direction": "Lower"
    },
    {
        "Metric": "Train–unseen normalized Wasserstein distance",
        "Value": generalisation_metrics_row[
            "log_likelihood_normalized_wasserstein_distance"
        ],
        "Preferred direction": "Lower"
    },
    {
        "Metric": "Train–unseen posterior-confidence gap",
        "Value": generalisation_metrics_row[
            "absolute_mean_maximum_posterior_gap"
        ],
        "Preferred direction": "Lower"
    },
    {
        "Metric": "Train–unseen cluster-proportion JS distance",
        "Value": cluster_proportion_js_distance,
        "Preferred direction": "Lower"
    }
])

paper_metrics_df.to_csv(
    os.path.join(
        NEW_OUTPUT_DIR,
        "simclr_512_recommended_paper_metrics.csv"
    ),
    index=False
)

paper_metrics_df.to_excel(
    os.path.join(
        NEW_OUTPUT_DIR,
        "simclr_512_recommended_paper_metrics.xlsx"
    ),
    index=False
)

print("\nTraining comparison metrics:")
print(
    comparison_metrics_df.to_string(
        index=False
    ),
    flush=True
)

print("\nTrain–unseen generalisation metrics:")
print(
    generalisation_metrics_df.to_string(
        index=False
    ),
    flush=True
)

print(
    "\nStandardized comparison files saved:",
    flush=True
)

print(
    "  simclr_512_model_comparison_metrics.csv/xlsx",
    flush=True
)

print(
    "  simclr_512_generalisation_comparison_metrics.csv/xlsx",
    flush=True
)

print(
    "  simclr_512_recommended_paper_metrics.csv/xlsx",
    flush=True
)

print(
    "  comparison_gmm_stability_runs.csv/xlsx",
    flush=True
)

# ==========================================================
# JOURNAL FIGURE — K=3 PCA AND t-SNE SIDE BY SIDE
# ==========================================================
# This block uses variables already calculated earlier:
#
# labels   = original GMM assignments
# emb_pca  = two-dimensional PCA coordinates
# emb_tsne = two-dimensional t-SNE coordinates
# var      = PCA explained variance percentages
#
# Original K=3 cluster positions:
#   Cluster 0 = upper-left radial-associated population
#   Cluster 1 = lower-left radial-associated population
#   Cluster 2 = right-hand mosaic-associated population
#
# Plot colours:
#   Cluster 2 = blue   = mosaic-associated state
#   Cluster 0 = orange = radial-associated state I
#   Cluster 1 = yellow = radial-associated state II
#
# IMPORTANT:
# The original numerical cluster labels are NOT modified.
# Only their plotting colours and displayed names are changed.
# ==========================================================

print(
    "\nCreating combined K=3 PCA and t-SNE figure...",
    flush=True
)


# ----------------------------------------------------------
# CHECK REQUIRED VARIABLES
# ----------------------------------------------------------

required_variables = {
    "labels": labels,
    "emb_pca": emb_pca,
    "emb_tsne": emb_tsne,
    "var": var
}

for variable_name, variable_value in required_variables.items():

    if variable_value is None:
        raise ValueError(
            f"Required variable '{variable_name}' is not available."
        )


if len(labels) != emb_pca.shape[0]:
    raise ValueError(
        "The number of GMM labels does not match the "
        "number of PCA coordinates."
    )

if len(labels) != emb_tsne.shape[0]:
    raise ValueError(
        "The number of GMM labels does not match the "
        "number of t-SNE coordinates."
    )

if len(np.unique(labels)) != 3:
    raise ValueError(
        "This figure requires exactly three GMM clusters, "
        f"but found labels: {np.unique(labels)}"
    )


# ----------------------------------------------------------
# CLUSTER DISPLAY SETTINGS
# ----------------------------------------------------------
# Dictionary keys are the ORIGINAL GMM cluster IDs.
#
# Based on the current PCA and t-SNE plots:
#   Original cluster 2 occupies the right-hand side and is
#   therefore displayed as the mosaic-associated population.

cluster_plot_settings = {
    2: {
        "name": "Mosaic-associated state",
        "colour": "#0072B2"       # blue
    },
    0: {
        "name": "Radial-associated state I",
        "colour": "#D55E00"       # orange
    },
    1: {
        "name": "Radial-associated state II",
        "colour": "#F0E442"       # yellow
    }
}

# Plotting order controls both point layering and legend order.
cluster_plot_order = [
    2,  # Mosaic-associated state
    0,  # Radial-associated state I
    1   # Radial-associated state II
]


# ----------------------------------------------------------
# PUBLICATION-STYLE SETTINGS
# ----------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ----------------------------------------------------------
# CREATE COMBINED FIGURE
# ----------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.0, 5.4)
)


# ==========================================================
# PANEL (a) — PCA
# ==========================================================

ax_pca = axes[0]

for original_cluster_id in cluster_plot_order:

    cluster_mask = (
        labels == original_cluster_id
    )

    settings = cluster_plot_settings[
        original_cluster_id
    ]

    ax_pca.scatter(
        emb_pca[cluster_mask, 0],
        emb_pca[cluster_mask, 1],
        s=8,
        color=settings["colour"],
        alpha=0.72,
        edgecolors="none",
        rasterized=True,
        label=settings["name"],
        zorder=2
    )


ax_pca.set_xlabel(
    f"PC1 ({var[0]:.1f}% explained variance)",
    labelpad=6
)

ax_pca.set_ylabel(
    f"PC2 ({var[1]:.1f}% explained variance)",
    labelpad=6
)

# No internal title.
ax_pca.set_title("")

# Panel label positioned like the previous journal figure.
ax_pca.text(
    -0.12,
    1.02,
    "(a)",
    transform=ax_pca.transAxes,
    fontsize=12,
    fontweight="normal",
    ha="left",
    va="bottom",
    clip_on=False
)


# ==========================================================
# PANEL (b) — t-SNE
# ==========================================================

ax_tsne = axes[1]

for original_cluster_id in cluster_plot_order:

    cluster_mask = (
        labels == original_cluster_id
    )

    settings = cluster_plot_settings[
        original_cluster_id
    ]

    ax_tsne.scatter(
        emb_tsne[cluster_mask, 0],
        emb_tsne[cluster_mask, 1],
        s=8,
        color=settings["colour"],
        alpha=0.72,
        edgecolors="none",
        rasterized=True,
        label=settings["name"],
        zorder=2
    )


ax_tsne.set_xlabel(
    "t-SNE 1",
    labelpad=6
)

ax_tsne.set_ylabel(
    "t-SNE 2",
    labelpad=6
)

# No internal title.
ax_tsne.set_title("")

ax_tsne.text(
    -0.12,
    1.02,
    "(b)",
    transform=ax_tsne.transAxes,
    fontsize=12,
    fontweight="normal",
    ha="left",
    va="bottom",
    clip_on=False
)


# ----------------------------------------------------------
# AXIS APPEARANCE
# ----------------------------------------------------------

for axis in axes:

    axis.grid(False)

    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4.5,
        width=0.9,
        colors="black"
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    axis.margins(
        x=0.025,
        y=0.035
    )


# ----------------------------------------------------------
# ONE LEGEND ONLY
# ----------------------------------------------------------

legend = ax_tsne.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#AFAFAF",
    framealpha=0.95,
    fontsize=8.5,
    borderpad=0.45,
    labelspacing=0.35,
    handletextpad=0.45,
    handlelength=1.1,
    scatterpoints=1
)

legend.get_frame().set_linewidth(
    0.8
)


# ----------------------------------------------------------
# PANEL SPACING
# ----------------------------------------------------------

fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.13,
    top=0.96,
    wspace=0.22
)


# ----------------------------------------------------------
# SAVE PNG AND PDF
# ----------------------------------------------------------

combined_figure_base = os.path.join(
    NEW_OUTPUT_DIR,
    "simclr_gmm_K3_PCA_tSNE_combined_journal"
)

fig.savefig(
    combined_figure_base + ".png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    combined_figure_base + ".pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print(
    "Saved combined K=3 PCA and t-SNE figure:",
    combined_figure_base,
    flush=True
)