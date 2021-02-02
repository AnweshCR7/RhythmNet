# import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.data_parser import preprocess_video_to_frame, read_target_data, calculate_hr

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, st_maps_path, target_signal_path):
        self.H = 125
        self.W = 125
        self.C = 3
        # self.video_path = data_path
        self.st_maps_path = st_maps_path
        # self.resize = resize
        self.target_path = target_signal_path
        self.maps = None

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
        return len(self.st_maps_path)

    def __getitem__(self, index):
        # identify the name of the video file so as to get the ground truth signal
        self.video_file_name = self.st_maps_path[index].split('/')[-1].split('.')[0]
        targets, timestamps = read_target_data(self.target_path, self.video_file_name)
        # sampling rate is video fps (check)
        target_hr = [calculate_hr(targets, timestamps=timestamps)]

        # Load the maps for video at 'index'
        self.maps = np.load(self.st_maps_path[index])
        shape = self.maps.shape
        self.maps = self.maps.reshape((-1, shape[3], shape[1], shape[2]))

        return {
            "st_maps": torch.tensor(self.maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }
