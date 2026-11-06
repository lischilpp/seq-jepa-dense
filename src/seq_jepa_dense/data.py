"""PathoROB Tolkach ESCA with random multi-crop sampling (paper, Sections 3.1-3.2)."""
import math

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from datasets import load_from_disk
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

N_FOLDS = 5     # stratified (class) group (slide_id) split; fold 0 is the test set
SPLIT_SEED = 42


def split_indices(labels, slide_ids):
    """Train and test indices of the slide-level split used in the paper."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(labels)), labels, slide_ids))
    return np.sort(train_idx), np.sort(test_idx)


class TolkachESCA(Dataset):
    """Each item holds `num_crops` random square crops of one 256x256 px image,
    resized to `input_size`, and their coordinates (u, v, zeta): the crop center's
    row and column, normalized by the image size, and the log-zoom
    log(input_size / crop side length). Actions are coordinate differences
    between successive crops and are computed by the model.

    split: "train" applies photometric augmentation to every crop. "test" and
    "all" (train + test, for visualization) are unaugmented, and their crops are
    seeded per image so that every model is evaluated on identical crops.
    """

    def __init__(self, data_path, split, num_crops, input_size=128, seed=42, scale=(0.08, 1.0),
                 color_jitter=(0.4, 0.4, 0.4, 0.1), color_jitter_p=0.8,
                 grayscale_p=0.2, blur_sigma=(0.1, 2.0), blur_p=0.5):
        assert split in ("train", "test", "all")
        ds = load_from_disk(data_path)
        labels = np.array(ds["biological_class"])
        self.classes = sorted(set(labels.tolist()))

        if split == "all":
            idx = np.arange(len(labels))
        else:
            train_idx, test_idx = split_indices(labels, np.array(ds["slide_id"]))
            idx = train_idx if split == "train" else test_idx
        self.data = ds.select(idx.tolist())
        self.labels = np.array([self.classes.index(c) for c in labels[idx]])

        self.split = split
        self.num_crops = num_crops
        self.input_size = input_size
        self.seed = seed
        self.scale = scale

        if split == "train":
            # 25x25 px blur kernel for 128 px crops
            kernel_size = input_size // 10 * 2 + 1
            self.augment = T.Compose([
                T.RandomApply([T.ColorJitter(*color_jitter)], p=color_jitter_p),
                T.RandomGrayscale(p=grayscale_p),
                T.RandomApply([T.GaussianBlur(kernel_size, sigma=blur_sigma)], p=blur_p),
            ])
        else:
            self.augment = torch.nn.Identity()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(img):
        """[0, 1] -> [-1, 1]"""
        return img * 2 - 1

    def _sample_crops(self, img):
        height, width = img.shape[-2:]
        crops, coords = [], []
        for _ in range(self.num_crops):
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=self.scale, ratio=(1.0, 1.0))
            crop = TF.resized_crop(img, i, j, h, w, size=[self.input_size, self.input_size],
                                   interpolation=T.InterpolationMode.BICUBIC)
            crops.append(self.normalize(self.augment(crop)))
            coords.append(((i + h / 2) / height, (j + w / 2) / width, math.log(self.input_size / h)))
        return torch.stack(crops), torch.tensor(coords, dtype=torch.float32)

    def __getitem__(self, idx):
        img = TF.pil_to_tensor(self.data[idx]["image"].convert("RGB")).float().div(255)
        if self.split == "train":
            crops, coords = self._sample_crops(img)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + idx)
                crops, coords = self._sample_crops(img)
        return {"label": int(self.labels[idx]), "crops": crops, "coords": coords}


def make_loader(dataset, batch_size, num_workers, seed, shuffle=False, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, generator=generator, pin_memory=True,
                      persistent_workers=num_workers > 0)
