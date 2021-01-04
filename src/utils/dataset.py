import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        self.augmentation_pipeline = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Image has 4 channels -> converting to RGB
        image = Image.open(self.image_paths[index]).convert("RGB")
        targets = self.targets[index]

        if self.resize is not None:
            # write as HxW
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        # convert to numpy array
        image = np.array(image)
        augmented = self.augmentation_pipeline(image=image)
        image = augmented['image']

        # Convert to form: CxHxW
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }