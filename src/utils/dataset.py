# import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.data_parser import preprocess_video_to_frame, read_target_data, get_hr

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, data_path, target_signal_path, num_frames=3000, resize=None, clip_size=300):
        self.H = 125
        self.W = 125
        self.C = 3
        self.time_depth = num_frames
        # self.video_path = data_path
        self.st_map_path = data_path
        # self.resize = resize
        self.target_path = target_signal_path
        self.frames = None
        self.clip_size = clip_size
        self.num_slices = int(self.time_depth/self.clip_size)

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
        return 1

    def __getitem__(self, index):
        self.video_file_name = self.st_map_path.split('/')[-1].split('.')[0]
        # self.frames = preprocess_video_to_frame(self.video_path, self.time_depth, (self.H, self.W), index, self.clip_size)
        self.frames = np.load(self.st_map_path)
        shape = self.frames.shape
        self.frames = self.frames.reshape((-1, shape[3], shape[1], shape[2]))

        targets = read_target_data(self.target_path, self.video_file_name)
        # sampling rate is video fps (check)
        target_hr = get_hr(targets, sampling_rate=50)

        return {
            "frame": torch.tensor(self.frames, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }
