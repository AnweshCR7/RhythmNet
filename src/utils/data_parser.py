import cv2
import os
import numpy as np
import torch
import pickle
from scipy import signal
import heartpy as hp
from tqdm import tqdm


def preprocess_video_to_frame(video_path, time_depth, output_shape):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate

    # print(f"parsing file: {video_path} at {frameRate} FPS")
    # face_detect = cv2.CascadeClassifier(os.path.join(os.getcwd(), face_detector_cascade_path))

    frame_count = 0
    faces = []
    # Initialize frames with zeros
    frames = np.zeros((time_depth, 36, 36, 3))
    start_frame = 0
    end_frame = time_depth

    '''
       Preprocess the Image
       Step 1: Use cv2 face detector based on Haar cascades
       Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
       Step 3: Downsample the face cropped frame to output_shape = 36x36
   '''

    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        # if frameId % math.floor(frameRate) == 0:
        # considering a 30 second clip: FPS x time => 50 * 30 = 1500 frames
        if start_frame < frameId < end_frame:
            # if not len(faces):
            #     faces = face_detect.detectMultiScale(frame, 1.3, 5)
            #
            # if len(faces) > 0:
            #     (x, y, w, h) = faces[0]
            #     cropped_face_frame = frame[y:y + h, x:x + w]
            #
            # Downsample to 36x36 using bicubic interpolation and rename cropped frame to frame
            try:
                frame = cv2.resize(frame, output_shape, interpolation=cv2.INTER_CUBIC)
            except:
                print('\n--------- ERROR! -----------\nUsual cv empty error')
                print(f'Shape of img1: {frame.shape}')
                # print(f'bbox: {bbox}')
                print(f'This is at idx: {frameId}')
                exit(666)
            # # convert to pytorch tensor
            frame = torch.from_numpy(frame)
            # HxWxC -> CxHxW
            frame = frame.permute(2, 0, 1)

            # filename = f"framess{frame_count}.jpg"
            frames[frame_count, :, :, :] = frame
            frame_count += 1

    cap.release()
    return frames


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
