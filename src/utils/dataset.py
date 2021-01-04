import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.data_parser import preprocess_video_to_frame, read_target_data

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """

    def __init__(self, video_path, target_path, num_frames=3000, resize=None):
        self.H = 36
        self.W = 36
        self.C = 3
        self.time_depth = num_frames
        self.video_path = video_path
        # self.resize = resize
        self.video_file_name = video_path.split('/')[-1].split('.')[0]
        self.target_path = target_path
        self.frames = None

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # self.augmentation_pipeline = albumentations.Compose(
        #     [
        #         albumentations.Normalize(
        #             mean, std, max_pixel_value=255.0, always_apply=True
        #         )
        #     ]
        # )

    def __len__(self):
        return self.time_depth

    def __getitem__(self, index):
        # Check if frames have been stored already
        if self.frames is None:
            self.frames = preprocess_video_to_frame(self.video_path, self.time_depth, (self.H, self.W))

        # # Image has 4 channels -> converting to RGB
        # image = Image.open(self.image_paths[index]).convert("RGB")
        targets = read_target_data(self.target_path, self.video_file_name)[index]

        # if self.resize is not None:
        #     # write as HxW
        #     image = image.resize(
        #         (self.resize[1], self.resize[0]), resample=Image.BILINEAR
        #     )
        #
        # # convert to numpy array
        # image = np.array(image)
        # augmented = self.augmentation_pipeline(image=image)
        # image = augmented['image']
        #
        # # Convert to form: CxHxW
        # image = np.transpose(image, (2,0,1)).astype(np.float32)

        return {
            "frame": torch.tensor(self.frames[index], dtype=torch.float),
            "target": torch.tensor(targets, dtype=torch.long)
        }
