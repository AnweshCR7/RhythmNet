import cv2
import os
import torch
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


def get_spatio_temporal_map():
    maps = np.zeros((10, 300, 25, 3))
    for index in range(1):
        # print(index)
        maps[index, :, :, :] = preprocess_video_to_frame(
            video_path="/Users/anweshcr7/github/RhythmNet/data/face_video/s01_trial01.avi", time_depth=300,
            output_shape=(125, 125), index=index, clip_size=300)

    # np.save('sp_maps.npy', maps)
    return maps


def preprocess_video_to_frame(video_path, time_depth, output_shape, index, clip_size):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate

    frame_count = 0
    # Initialize frames with zeros
    frames = np.zeros((time_depth, output_shape[0], output_shape[1], 3))
    start_frame = index * clip_size
    end_frame = start_frame + clip_size
    detector = get_haarcascade()
    spatio_temporal_map = np.zeros((clip_size, 25, 3))

    '''
       Preprocess the Image
       Step 1: Use cv2 face detector based on Haar cascades
       Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
       Step 3: Downsample the face cropped frame to output_shape = 36x36
   '''

    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(frame)
        # We expect only one face
        if len(faces) is not 0:
            (x, y, w, d) = faces[0]
        else:
            continue
        # overlay rectangle as per detected face.
        # cv2.rectangle(frame, (x, y), (x + w, y + d), (255, 255, 255), 2)
        frame_cropped = frame[y:(y + d), x:(x + w)]
        frame_resized = np.zeros((output_shape[0], output_shape[1], 3))

        if not ret:
            break
        # considering a 30 second clip: FPS x time => 50 * 30 = 1500 frames
        if start_frame < frameId < end_frame:
            window_name = 'image'
            # Downsample to 36x36 using bicubic interpolation and rename cropped frame to frame
            # try:
            frame_resized = cv2.resize(frame_cropped, output_shape, interpolation=cv2.INTER_CUBIC)
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)

            # except:
            #     print('\n--------- ERROR! -----------\nUsual cv empty error')
            #     print(f'Shape of img1: {frame.shape}')
            #     # print(f'bbox: {bbox}')
            #     print(f'This is at idx: {frameId}')
            #     exit(666)

            # filename = f"framess{frame_count}.jpg"
            # plot_image(frame_resized)
            roi_blocks = chunkify(frame_resized)
            for block_idx, block in enumerate(roi_blocks):
                # plot_image(block)
                # start from zero index
                # block_idx -= 1
                avg_pixels = cv2.mean(block)
                spatio_temporal_map[frame_count, block_idx, 0] = avg_pixels[0]
                spatio_temporal_map[frame_count, block_idx, 1] = avg_pixels[1]
                spatio_temporal_map[frame_count, block_idx, 2] = avg_pixels[2]
            frames[frame_count, :, :, :] = frame_resized
            frame_count += 1

        # might not be necessary
        if frame_count == clip_size - 1:
            break

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
    ppg_data = np.array(get_ppg_channel(x["data"][trial_number]))
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
    wd_data, m_data = hp.process(signal_data, sample_rate=sampling_rate, high_precision=True, clean_rr=True)

    return [m_data["bpm"]]


def make_csv():
    video_file_paths = glob.glob(config.ST_MAPS_PATH + "/*.npy")
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
    # data = get_spatio_temporal_map()
    make_csv()
    print('done')