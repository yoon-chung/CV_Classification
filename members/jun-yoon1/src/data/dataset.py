from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(
        self,
        ids: list[str],
        image_dir: str | Path,
        targets: list[int] | None,
        transform,
        strong_transform=None,
        strong_classes: set[int] | None = None,
    ):
        self.ids = ids
        self.image_dir = Path(image_dir)
        self.targets = targets
        self.transform = transform
        self.strong_transform = strong_transform
        self.strong_classes = strong_classes or set()

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_id: str):
        image_path = self.image_dir / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image = self._load_image(image_id)

        if self.targets is None:
            out = self.transform(image=image)
            return out["image"], image_id

        target = int(self.targets[idx])
        if self.strong_transform is not None and target in self.strong_classes:
            out = self.strong_transform(image=image)
        else:
            out = self.transform(image=image)

        return out["image"], torch.tensor(target, dtype=torch.long), image_id
