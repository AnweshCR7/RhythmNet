import cv2
import os
import pickle
import glob
import numpy as np
import pandas as pd
import src.config as config
from scipy import signal
import heartpy as hp
from tqdm import tqdm
import matplotlib.pyplot as plt
# used for accessing url to download files
import urllib.request as urlreq
from sklearn import model_selection
from sklearn import preprocessing
from joblib import Parallel, delayed, parallel_backend
import time

# download requisite certificates
import ssl;

ssl._create_default_https_context = ssl._create_stdlib_context


def chunkify(img, block_width=5, block_height=5):
    shape = img.shape
    x_len = shape[0] // block_width
    y_len = shape[1] // block_height
    # print(x_len, y_len)

    chunks = []
    x_indices = [i for i in range(0, shape[0] + 1, block_width)]
    y_indices = [i for i in range(0, shape[1] + 1, block_height)]

    shapes = list(zip(x_indices, y_indices))

    for i in range(len(shapes) - 1):
        # try:
        start_x = shapes[i][0]
        start_y = shapes[i][1]
        end_x = shapes[i + 1][0]
        end_y = shapes[i + 1][1]
        chunks.append(img[start_x:end_x, start_y:end_y])
        # except IndexError:
        #     print('End of Array')

    return chunks


def plot_image(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def get_haarcascade():
    haarcascade_url = config.haarcascade_url
    haarcascade_filename = haarcascade_url.split('/')[-1]
    # chech if file is in working directory
    if haarcascade_filename in os.listdir(os.curdir):
        # print("xml file already exists")
        pass
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
        urlreq.urlretrieve(haarcascade_url, haarcascade_filename)
        print("xml file downloaded")

    return cv2.CascadeClassifier(haarcascade_filename)


def get_spatio_temporal_map_threaded(file):
    # print(f"Generating Maps for file: {file}")
    maps = np.zeros((10, config.CLIP_SIZE, 25, 3))
    for index in range(2):
        # print(index)
        maps[index, :, :, :] = preprocess_video_to_frame(
            video_path=file,
            output_shape=(125, 125), slice_index=index, clip_size=config.CLIP_SIZE)

    file_name = file.split('/')[-1].split('.')[0]
    folder_name = file.split('/')[-2]
    save_path = os.path.join(config.ST_MAPS_PATH, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"{file_name}.npy")
    # np.save(f"{config.ST_MAPS_PATH}{file_name}.npy", maps)
    np.save(save_path, maps)
    return 1


def get_spatio_temporal_map_threaded_wrapper():
    video_files = glob.glob(config.FACE_DATA_DIR + '/**/*avi')


    # try:
    #     pool = Pool(3)  # on 8 processors
    #     pool.map(get_spatio_temporal_map_threaded, video_files[:4])
    # finally:  # To make sure processes are closed in the end, even if errors happen
    #     pool.close()
    #     pool.join()
    start = time.time()
    with parallel_backend("loky", inner_max_num_threads=4):
        Parallel(n_jobs=4)(delayed(get_spatio_temporal_map_threaded)(file) for file in tqdm(video_files))
    end = time.time()

    print('{:.4f} s'.format(end - start))
    # with multiprocessing.Pool(processes=4) as pool:
    #     results = list(tqdm(pool.starmap_async(get_spatio_temporal_map_threaded, product(video_files[:4])), total=len(video_files)))


def get_spatio_temporal_map():
    video_files = glob.glob(config.FACE_DATA_DIR + '/**/*avi')
    start = time.time()
    for file in tqdm(video_files[:2]):
        maps = np.zeros((10, config.CLIP_SIZE, 25, 3))
        for index in range(10):
            # print(index)
            maps[index, :, :, :] = preprocess_video_to_frame(
                video_path=file,
                output_shape=(125, 125), slice_index=index, clip_size=config.CLIP_SIZE)

        file_name = file.split('/')[-1].split('.')[0]
        folder_name = file.split('/')[-2]
        save_path = os.path.join(config.ST_MAPS_PATH, folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, f"{file_name}.npy")
        # np.save(f"{config.ST_MAPS_PATH}{file_name}.npy", maps)
        np.save(save_path, maps)

    end = time.time()
    print('{:.4f} s'.format(end - start))
    # return maps


def preprocess_video_to_frame(video_path, output_shape, slice_index, clip_size):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate
    min_max_scaler = preprocessing.MinMaxScaler()
    # Initialize frames with zeros
    frames = np.zeros((clip_size, output_shape[0], output_shape[1], 3))
    start_frame = slice_index * clip_size

    end_frame = start_frame + clip_size
    detector = get_haarcascade()

    # 25 needs to be a variable
    spatio_temporal_map = np.zeros((clip_size, 25, 3))

    '''
       Preprocess the Image
       Step 1: Use cv2 face detector based on Haar cascades
       Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
       Step 3: Downsample the face cropped frame to output_shape = 36x36
   '''
    # frame counter is our zero indexed counter for monitoring clip_size
    frame_counter = 0
    while cap.isOpened():
        curr_frame_id = int(cap.get(1))  # current frame number
        ret, frame = cap.read()

        if start_frame <= curr_frame_id < end_frame:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detectMultiScale(frame)
            # We expect only one face
            if len(faces) is not 0:
                (x, y, w, d) = faces[0]
            else:
                (x, y, w, d) = (308, 189, 217, 217)
            # overlay rectangle as per detected face.
            # cv2.rectangle(frame, (x, y), (x + w, y + d), (255, 255, 255), 2)
            frame_cropped = frame[y:(y + d), x:(x + w)]
            frame_resized = np.zeros((output_shape[0], output_shape[1], 3))

            if not ret:
                break

            window_name = 'image'
            # Downsample to 36x36 using bicubic interpolation and rename cropped frame to frame
            try:
                frame_resized = cv2.resize(frame_cropped, output_shape, interpolation=cv2.INTER_CUBIC)
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)

            except:
                print('\n--------- ERROR! -----------\nUsual cv empty error')
                print(f'Shape of img1: {frame.shape}')
                # print(f'bbox: {bbox}')
                print(f'This is at idx: {curr_frame_id}')
                exit(666)

            # filename = f"framess{frame_count}.jpg"
            # plot_image(frame_resized)
            roi_blocks = chunkify(frame_resized)
            for block_idx, block in enumerate(roi_blocks):
                # plot_image(block)
                avg_pixels = cv2.mean(block)
                spatio_temporal_map[frame_counter, block_idx, 0] = avg_pixels[0]
                spatio_temporal_map[frame_counter, block_idx, 1] = avg_pixels[1]
                spatio_temporal_map[frame_counter, block_idx, 2] = avg_pixels[2]

            frames[frame_counter, :, :, :] = frame_resized
            frame_counter += 1
        else:
            # ignore frames outside the clip
            pass

        # Necessary to break the outer while
        # Breaks when we have read the clip b/w start_frame till end_frame
        if frame_counter == (end_frame-start_frame):
            break

    for block_idx in range(spatio_temporal_map.shape[1]):
        # Not sure about uint8
        fn_scale_0_255 = lambda x: (x * 255.0).astype('uint8')
        scaled_channel_0 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 0].reshape(-1, 1))
        spatio_temporal_map[:, block_idx, 0] = fn_scale_0_255(scaled_channel_0.flatten())
        scaled_channel_1 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 1].reshape(-1, 1))
        spatio_temporal_map[:, block_idx, 1] = fn_scale_0_255(scaled_channel_1.flatten())
        scaled_channel_2 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 2].reshape(-1, 1))
        spatio_temporal_map[:, block_idx, 2] = fn_scale_0_255(scaled_channel_2.flatten())


    cap.release()
    return spatio_temporal_map


def get_ppg_channel(x):
    # i think PPG channel is at 38
    return x[38]


def read_target_data(target_data_path, video_file_name):
    # print(f'reading the PPG signal for video: {video_file_name}')
    dat_file_name = video_file_name.split('_')[0]
    trial_number = int(video_file_name.split('_')[1][-2:])

    x = pickle.load(open(os.path.join(os.getcwd(), target_data_path, f"{dat_file_name}.dat"), 'rb'), encoding='latin1')
    ppg_data = np.array(get_ppg_channel(x["data"][trial_number - 1]))
    # return signal.resample(ppg_data, 3000)

    return filter_and_resample_signal(signal_data=ppg_data, resampling_dim=3000)


def filter_and_resample_signal(signal_data, resampling_dim):
    # video sample rate is given to be 128
    sample_rate = 128
    filtered = hp.filter_signal(signal_data, [0.7, 2.5], sample_rate=sample_rate,
                                order=3, filtertype='bandpass')
    filtered = signal.resample(filtered, resampling_dim)

    return filtered


def get_hr(signal_data, sampling_rate):
    # High precision gives an error in signal after re-sampling
    # wd_data, m_data = hp.process(signal_data, sample_rate=sampling_rate, high_precision=True, clean_rr=True)
    wd_data, m_data = hp.process(signal_data, sample_rate=sampling_rate)

    return [m_data["bpm"]]


def make_csv():
    video_file_paths = glob.glob(config.ST_MAPS_PATH + "/**/*.npy")[:6]
    video_files = []
    for path in video_file_paths:
        split_by_path = path.split('/')
        video_file = os.path.join(split_by_path[-2], split_by_path[-1])
        video_files.append(video_file)

    num_folds = 2
    kf = model_selection.KFold(n_splits=num_folds)

    col_names = ['video', 'set', 'iteration']
    df = pd.DataFrame(columns=col_names)

    fold = 1
    for train_idx, validation_idx in kf.split(video_files):
        trainDF = pd.DataFrame([video_files[idx] for idx in train_idx], columns=['video'])
        validateDF = pd.DataFrame([video_files[idx] for idx in validation_idx], columns=['video'])
        trainDF[['set', 'iteration']] = 'T', fold
        validateDF[['set', 'iteration']] = 'V', fold
        fold += 1

        df = pd.concat([df, trainDF, validateDF])
        df.to_csv(config.SAVE_CSV_PATH, index=False)

    return


if __name__ == '__main__':
    # maps = np.zeros((10, 300, 25, 3))
    # for index in range(10):
    #     # print(index)
    #     maps[index, :, :, :] = preprocess_video_to_frame(
    #         video_path="/Users/anweshcr7/github/RhythmNet/data/face_video/s01_trial01.avi", time_depth=300,
    #         output_shape=(125, 125), index=index, clip_size=300)
    #
    # np.save('sp_maps.npy', maps)
    # get_spatio_temporal_map()
    # get_spatio_temporal_map_threaded_wrapper()
    # video_files = glob.glob(config.FACE_DATA_DIR + '/**/*avi')
    # r = list(process_map(get_spatio_temporal_map_threaded, video_files[:2], max_workers=1))
    # signal = read_target_data("/Users/anweshcr7/github/RhythmNet/data/data_preprocessed/", "s01_trial04")
    # get_hr(signal, 50)

    make_csv()
    print('done')