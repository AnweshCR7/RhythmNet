# FACE_DATA_DIR = "/content/drive/MyDrive/data/deep_phys/face_videos/"
# DATA_PATH = "/content/drive/MyDrive/data/rhythmnet/st_maps/"
# TARGET_SIGNAL_DIR = "/content/drive/MyDrive/data/deep_phys/data_preprocessed/"
# SAVE_CSV_PATH = "/content/drive/MyDrive/data/rhythmnet/kfold.csv"
# ST_MAPS_PATH = "/content/drive/MyDrive/data/rhythmnet/st_maps/"
# CHECKPOINT_PATH = "/content/drive/MyDrive/data/rhythmnet/checkpoint"
# PLOT_PATH = "/content/drive/MyDrive/data/rhythmnet/plots"
# NUM_WORKERS = 2
# DEVICE = "cuda"
# BATCH_SIZE = 10
# EPOCHS = 50
# lr = 1e-3
# CLIP_SIZE = 300

# For INSY server

# FACE_DATA_DIR = "/content/drive/MyDrive/data/deep_phys/face_videos/"
# HOME_DIR = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/amarwade/"
# HR_DATA_PATH = HOME_DIR + "data/ECG_Fitness/ecg_hr_csv/"
# DATA_PATH = HOME_DIR + "data/ECG_Fitness/ecg_st_maps/"
# TARGET_SIGNAL_DIR = HOME_DIR + "data/DEAP/data_preprocessed/"
# SAVE_CSV_PATH = HOME_DIR + "RhythmNet/ecg_subject_exclusive_folds.csv"
# ST_MAPS_PATH = HOME_DIR + "data/ECG_Fitness/ecg_st_maps/"
# CHECKPOINT_PATH = HOME_DIR + "checkpoints/RhythmNet"
# PLOT_PATH = HOME_DIR + "plots/RhythmNet"
# NUM_WORKERS = 2
# DEVICE = "cuda"
# BATCH_SIZE = 16
# EPOCHS = 20
# lr = 1e-3
# CLIP_SIZE = 300
# TENSORBOARD_PATH = HOME_DIR + "/runs/"
# GRU_TEMPORAL_WINDOW = 6

haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
eye_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
# FACE_DATA_DIR = "../data/face_video/"
# HR_DATA_PATH = "../data/VIPL_hr_csv/"
HR_DATA_PATH = "../data/VIPL_hr_csv/"
FACE_DATA_DIR = "/Volumes/T7/ecg_st_maps/"
TARGET_SIGNAL_DIR = "/Users/anweshcr7/Downloads/CleanerPPG/ECG-Fitness/Cleaned/"
# SAVE_CSV_PATH = "subject_exclusive_folds.csv"
SAVE_CSV_PATH = "ecg_subject_exclusive_folds.csv"
ST_MAPS_PATH = "/Volumes/T7/ecg_npy/"
# ST_MAPS_PATH = "/Volumes/Backup Plus/vision/DEAP_emotion/st_maps/"
CHECKPOINT_PATH = "../checkpoint"
DATA_PATH = "../data/"
PLOT_PATH = "../plots"
BATCH_SIZE = 16
EPOCHS = 10
EPOCHS_TEST = 1
CLIP_SIZE = 300
lr = 1e-3
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 0
DEVICE = "cpu"
GRU_TEMPORAL_WINDOW = 6