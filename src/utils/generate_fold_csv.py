import glob
import pandas as pd
import os
import cv2
from sklearn import model_selection
import scipy.io


def preprocess_file_name(file_path):
    split_by_path = file_path.split('/')
    preprocessed_file_name = "_".join(split_by_path[-4:-1])
    return os.path.join("vipl_npy", f"{preprocessed_file_name}.npy")


def make_csv(fold_data_dict):
    # video_file_paths = glob.glob(config.ST_MAPS_PATH + "/**/*.npy")


    # video_file_paths = glob.glob("/Users/anweshcr7/thesis/src/data/vipl_npy/*.npy")
    # video_files = []
    #
    # for path in video_file_paths:
    #     split_by_path = path.split('/')
    #     video_file = os.path.join(split_by_path[-2], split_by_path[-1])
    #     video_files.append(video_file)
    #
    # video_files = [x for x in video_files if "source4" not in x]
    # num_folds = 5
    # kf = model_selection.KFold(n_splits=num_folds)

    col_names = ['video', 'fold']
    df = pd.DataFrame(columns=col_names)

    fold = 1

    for idx, fold in enumerate(fold_data_dict.keys()):
        video_files_fold = []
        fold_subjects = [str(x) for x in fold_data_dict[fold].squeeze(0)]
        for subject in fold_subjects:
            video_files_fold.extend(glob.glob(f"/Volumes/Backup Plus/vision/VIPL-HR/data/*/p{subject}/*/*/*.avi"))

        # Don't consider NIR videos
        video_files_fold = [file_path for file_path in video_files_fold if "source4" not in file_path]
        video_files_fold = [preprocess_file_name(file_path) for file_path in video_files_fold]


        trainDF = pd.DataFrame(video_files_fold, columns=['video'])
        trainDF['fold'] = idx + 1

        df = pd.concat([df, trainDF])
        df.to_csv("VIPL_folds_final.csv", index=False)

    print("done")


    # for train_idx, validation_idx in kf.split(video_files):
    #     trainDF = pd.DataFrame([video_files[idx] for idx in train_idx], columns=['video'])
    #     validateDF = pd.DataFrame([video_files[idx] for idx in validation_idx], columns=['video'])
    #     trainDF[['set', 'iteration']] = 'T', fold
    #     validateDF[['set', 'iteration']] = 'V', fold
    #     fold += 1
    #
    #     df = pd.concat([df, trainDF, validateDF])
    #     df.to_csv("VIPL_npy.csv", index=False)

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
    fold_data_dict = {}
    fold_files = glob.glob("/Volumes/Backup Plus/vision/VIPL-HR/fold/*.mat")
    for fold in fold_files:
        name = fold.split('/')[-1].split('.')[0]
        fold_data = scipy.io.loadmat(fold)
        fold_data_dict[name] = fold_data[name]
    make_csv(fold_data_dict)
    print("done")