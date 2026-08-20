import io
import os
import random
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from leap.determinism import stable_slide_seed

PATCH_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CytologyTransform:
    """Resize a patch to PATCH_SIZE and normalise with ImageNet statistics."""

    def __init__(self, patch_size: int = PATCH_SIZE):
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img):
        return self.transform(img)


class AugmentationTransform:
    """Flip, jitter colour, then resize and normalise. Used only when augment=True."""

    def __init__(self, patch_size: int = PATCH_SIZE):
        self.augment = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
            ]),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img):
        return self.augment(img)


class CytologyDataset(Dataset):
    """One bag of single-cell patches per slide, read from `<image_folder>/<Slide_ID>.zip`.

    Parameters
    ----------
    label_file: spreadsheet with a Slide_ID column and `label_column`.
    image_folder: directory of per-slide zip archives of PNG patches.
    label_column: column holding the slide label.
    patches_per_slide: how many patches to draw per slide per bag.
    transform: patch transform; defaults to CytologyTransform().
    augment: append an augmented copy of every drawn patch, doubling bag size.
    index_list: positional row indices of `label_file` to restrict the dataset to.
    base_seed: seed for the per-slide deterministic draw.
    deterministic: if True, patches are drawn with a per-slide seeded RNG so a slide always
        yields the same patches (use for inference). If False, patches are re-drawn each
        epoch with Python's `random`, made reproducible by the dataloader's worker_init_fn
        (use for training).

    __getitem__ returns (patches, label): (N_PATCHES, C, H, W) float tensor and a scalar.
    Slides with fewer than `patches_per_slide` patches are zero-padded.
    """

    def __init__(
        self,
        label_file,
        image_folder,
        label_column,
        patches_per_slide,
        transform=None,
        augment=False,
        index_list=None,
        base_seed=42,
        deterministic=False,
    ):
        self.labels_df = pd.read_excel(label_file)
        if index_list is not None:
            self.labels_df = self.labels_df.iloc[index_list].reset_index(drop=True)

        self.image_folder = image_folder
        self.label_column = label_column
        self.patches_per_slide = patches_per_slide
        self.transform = transform if transform is not None else CytologyTransform()
        self.augment = augment
        self.base_seed = base_seed
        self.deterministic = deterministic
        if self.augment:
            self.augmentation_transform = AugmentationTransform()

        self.slide_ids = self.labels_df["Slide_ID"].tolist()
        self.selected_files = {}
        self.weights = self._class_balanced_weights()

    def get_label(self, slide_id):
        """The label of one slide, looked up by Slide_ID."""
        return self.labels_df.loc[
            self.labels_df["Slide_ID"] == slide_id, self.label_column
        ].values[0]

    def _class_balanced_weights(self):
        """Per-slide sampling weights inversely proportional to class frequency."""
        counts = self.labels_df[self.label_column].value_counts().to_dict()
        return [1.0 / counts[self.get_label(sid)] for sid in self.slide_ids]

    def class_counts(self):
        """label -> number of slides carrying it."""
        return self.labels_df[self.label_column].value_counts().to_dict()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        slide_id = self.labels_df.iloc[idx]["Slide_ID"]
        label = self.labels_df.iloc[idx][self.label_column]
        zip_path = os.path.join(self.image_folder, f"{slide_id}.zip")

        images = []
        with zipfile.ZipFile(zip_path, "r") as archive:
            patch_files = sorted(f for f in archive.namelist() if f.endswith(".png"))
            k = min(self.patches_per_slide, len(patch_files))
            if self.deterministic:
                rng = np.random.default_rng(stable_slide_seed(slide_id, self.base_seed))
                selected = list(rng.choice(patch_files, size=k, replace=False)) if k > 0 else []
            else:
                selected = random.sample(patch_files, k)
            self.selected_files[slide_id] = selected

            for patch_file in selected:
                with archive.open(patch_file) as handle:
                    img = Image.open(io.BytesIO(handle.read())).convert("RGB")
                images.append(self.transform(img))
                if self.augment:
                    images.append(self.augmentation_transform(img))

        target_size = self.patches_per_slide * (2 if self.augment else 1)
        while len(images) < target_size:
            images.append(torch.zeros_like(images[0]))

        return torch.stack(images), torch.tensor(label, dtype=torch.float32)

    def get_selected_files(self, slide_id):
        """The exact patch filenames drawn for one slide, or [] if it has not been read yet.

        Populated in the reading process, so it is only valid for loaders with num_workers=0.
        """
        return self.selected_files.get(slide_id, [])
