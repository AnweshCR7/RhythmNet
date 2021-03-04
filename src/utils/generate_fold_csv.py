import glob
import pandas as pd
import os
import cv2
from sklearn import model_selection


def make_csv():
    # video_file_paths = glob.glob(config.ST_MAPS_PATH + "/**/*.npy")
    video_file_paths = glob.glob("/Users/anweshcr7/thesis/src/data/vipl_npy/*.npy")
    video_files = []

    for path in video_file_paths:
        split_by_path = path.split('/')
        video_file = os.path.join(split_by_path[-2], split_by_path[-1])
        video_files.append(video_file)

    video_files = [x for x in video_files if "source4" not in x]
    num_folds = 5
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
        df.to_csv("VIPL_npy.csv", index=False)

    return


def make_csv_with_frame_rate():
    # video_file_paths = glob.glob(config.ST_MAPS_PATH + "/**/*.npy")
    video_file_paths = glob.glob("/Users/anweshcr7/thesis/src/data/vipl_npy/*.npy")
    video_source = "/Volumes/Backup Plus/vision/vipl_videos"
    video_files = []
    fr_dict = {}

    for path in video_file_paths:
        split_by_path = path.split('/')
        video_file = os.path.join(split_by_path[-2], split_by_path[-1])
        video_files.append(video_file)
        video_name = split_by_path[-1].split('.')[0] + ".avi"
        cap = cv2.VideoCapture(os.path.join(video_source, video_name))
        frameRate = cap.get(5)
        fr_dict[video_file] = frameRate
        cap.release()


    video_files = [x for x in video_files if "source4" not in x]
    num_folds = 5
    kf = model_selection.KFold(n_splits=num_folds)

    col_names = ['video', 'set', 'iteration', 'fps']
    df = pd.DataFrame(columns=col_names)

    fold = 1
    for train_idx, validation_idx in kf.split(video_files):
        trainDF = pd.DataFrame([video_files[idx] for idx in train_idx], columns=['video'])
        validateDF = pd.DataFrame([video_files[idx] for idx in validation_idx], columns=['video'])
        trainDF[['set', 'iteration']] = 'T', fold
        validateDF[['set', 'iteration']] = 'V', fold
        trainDF[['fps']] = [fr_dict[video_files[idx]] for idx in train_idx]
        validateDF[['fps']] = [fr_dict[video_files[idx]] for idx in validation_idx]
        fold += 1

        df = pd.concat([df, trainDF, validateDF])
        df.to_csv("VIPL_npy_with_fps.csv", index=False)

    return


if __name__ == '__main__':
    # make_csv_with_frame_rate()
    print("done")