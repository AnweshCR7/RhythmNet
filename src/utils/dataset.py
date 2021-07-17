# import albumentations
import torch
import h5py
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from utils.signal_utils import read_target_data, calculate_hr, get_hr_data
import matplotlib.pyplot as plt
import lmdb
from utils.FaceDataset import FaceDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


y_lmdb_path = "../data/ecg-fitness_face_linear-192x128_batch-300_test-train_y_lmdb"
X_lmdb_path = "../data/ecg-fitness_face_linear-192x128_batch-300_test-train_X_lmdb"


def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
    """
    # The data is now the SpatioTemporal Maps instead of videos
    # , y_lmdb_path, X_lmdb_path
    def __init__(self, st_maps_path, target_signal_path):
        self.transform = None
        self.H = 180
        self.W = 180
        self.C = 3
        self.train = True
        self.rgb = True
        # self.video_path = data_path
        self.st_maps_path = st_maps_path
        # self.resize = resize
        self.target_path = target_signal_path
        self.maps = None
        self.batch_size = 300
        self.label_env = lmdb.open(y_lmdb_path, readonly=True)
        self.data_env = lmdb.open(X_lmdb_path, readonly=True)

        self.data_txn = self.data_env.begin()
        self.data_cursor = self.data_txn.cursor()

        self.length = np.fromstring(self.data_cursor.get('frame_count'.encode('ascii')), dtype='int32')

        self.train_length = int(np.floor((self.length / self.batch_size) * (2.0 / 3.0)) * self.batch_size)
        # handle identity aware splitting
        while self.data_cursor.get('fps-{:08}'.format(self.train_length).encode('ascii'), default=None) is None:
            self.train_length += int(self.batch_size)

        self.test_length = self.length - self.train_length

        if self.train:
            self.length = self.train_length
            self.shift = 0
        else:
            self.length = self.test_length
            self.shift = self.train_length

        self.label_txn = self.label_env.begin()
        self.label_cursor = self.label_txn.cursor()

        FaceDataset.__init__(self, self.length, self.transform)
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


    def __get_height(self, cursor, index):
        search_index = int(index)
        string_data = cursor.get(('height-%08d' % search_index).encode('ascii'))
        while string_data is None:
            search_index -= int(self.batch_size)
            string_data = cursor.get(('height-%08d' % search_index).encode('ascii'))

            if search_index < 0:
                raise KeyError('Key cannot be lower than 0, started at %d...' % index)

        return int(np.fromstring(string_data, dtype='int32').round())

    def __get_data(self, cursor, index):
        key = '{:08}'.format(index)
        string_data = cursor.get(key.encode('ascii'))
        return string_data

    def get_fps_and_regularization_factor(self, idx):
        idx += self.shift

        fps_string = self.data_cursor.get('fps-{:08}'.format(idx).encode('ascii'), default=None)
        regularization_factor_string = self.data_cursor.get('regularization_factor-{:08}'.format(idx).encode('ascii'))
        while fps_string is None:
            idx -= int(self.batch_size)
            fps_string = self.data_cursor.get('fps-{:08}'.format(idx).encode('ascii'), default=None)
            regularization_factor_string = self.data_cursor.get('regularization_factor-{:08}'.format(idx).encode('ascii'))

        regularization_factor = float(np.fromstring(regularization_factor_string, dtype='float64').mean())

        return float(np.fromstring(fps_string, dtype='float64').mean()), regularization_factor

    def get_shift(self):
        return self.shift

    def get_original_and_transformed_im(self, idx=0):
        orig_data = self.get_im_data(idx)

        transformed_data = orig_data
        if self.transform is not None:
            transformed_data = self.do_transforms(orig_data)

        return orig_data.astype('uint8'), transformed_data.astype('uint8')

    def get_im_data(self, idx):
        string_data = self.__get_data(self.data_cursor, idx)
        if string_data is None:
            print('missing %d' % idx)

        height = self.__get_height(self.data_cursor, int(idx / self.batch_size) * int(self.batch_size))
        flat_data = np.fromstring(string_data, dtype='uint8')
        width = int((flat_data.size / 3) / height)
        flat_data = flat_data.reshape(height, width, 3).transpose((2, 1, 0))
        if not self.rgb:
            flat_data = flat_data[1, :, :].reshape(1, width, height)

        return flat_data.astype('float32')


    def __len__(self):
        return len(self.st_maps_path)

    def __getitem__(self, index):
        data = self.get_im_data(index)
        if self.transform is not None:
            data = self.do_transforms(data)

        string_data = self.__get_data(self.label_cursor, index)

        if string_data is not None:
            y = np.fromstring(string_data, dtype='int32')
            target = y[0].astype('float32').reshape(1,)
        else:
            raise ValueError('Missing ground truth in the database for key %d...' % index)

        return torch.tensor(data, dtype=torch.float), torch.tensor((target / 60.0), dtype=torch.float)




        # # identify the name of the video file so as to get the ground truth signal
        # self.video_file_name = self.st_maps_path[index].split('/')[-1].split('.')[0]
        # # targets, timestamps = read_target_data(self.target_path, self.video_file_name)
        # # sampling rate is video fps (check)
        # db = h5py.File(self.st_maps_path[index], 'r')
        # frames = db['frames']
        #
        # frames = frames[:10,:,:,:]
        # # Load the maps for video at 'index'
        # # self.maps = np.load(self.st_maps_path[index])
        # # map_shape = self.maps.shape
        # # self.maps = self.maps.reshape((-1, map_shape[3], map_shape[1], map_shape[2]))
        #
        #
        #
        # # target_hr = calculate_hr(targets, timestamps=timestamps)
        # # target_hr = calculate_hr_clip_wise(map_shape[0], targets, timestamps=timestamps)
        # # target_hr = get_hr_data(self.video_file_name)
        # print('hello')
        # # To check the fact that we dont have number of targets greater than the number of maps
        # # target_hr = target_hr[:map_shape[0]]
        # # self.maps = self.maps[:target_hr.shape[0], :, :, :]
        # return torch.tensor(frames, dtype=torch.float)
        # # return {
        # #     "video_file_name": self.video_file_name,
        # #     "st_maps": torch.tensor(frames, dtype=torch.float),
        # #     # "target": torch.tensor(target_hr, dtype=torch.float)
        # # }
