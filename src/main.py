import os
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import engine
import config
from utils.dataset import DataLoaderRhythmNet
from utils.plot_scripts import plot_train_test_curves
from utils.model_utils import plot_loss, load_model_if_checkpointed, save_model_checkpoint
from models.simpleCNN import SimpleCNN
from models.lenet import LeNet
from models.rhythmNet import RhythmNet


def run_training():

    # check path to checkpoint directory
    if config.CHECKPOINT_PATH:
        if not os.path.exists(config.CHECKPOINT_PATH):
            os.makedirs(config.CHECKPOINT_PATH)
            print("Output directory is created")

    # --------------------------------------
    # Initialize Model
    # --------------------------------------

    model = RhythmNet()

    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(42)
    else:
        print("GPU not available, using CPU")

    if config.CHECKPOINT_PATH:
        checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT_PATH)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("Output directory is created")

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.8, patience=5, verbose=True
    # )
    loss_fn = nn.L1Loss()

    # --------------------------------------
    # Build Dataloaders
    # --------------------------------------

    testset = trainset = None
    videos = glob.glob(config.FACE_DATA_DIR + '*.avi')
    st_maps = glob.glob(config.ST_MAPS_PATH + '*.npy')

    # Read from a pre-made csv file that contains data divided into folds for cross validation
    folds_df = pd.read_csv(config.SAVE_CSV_PATH)

    # Loop for enumerating through folds.
    print(f"Details: {len(folds_df['iteration'].unique())} fold training for {config.EPOCHS} Epochs (each video)")
    for k in folds_df['iteration'].unique():
        # Filter DF
        video_files_test = folds_df.loc[(folds_df['iteration'] == k) & (folds_df['set'] == 'V')]
        video_files_train = folds_df.loc[(folds_df['iteration'] == k) & (folds_df['set'] == 'T')]

        # Get paths from filtered DF
        video_files_test = [os.path.join(config.ST_MAPS_PATH, video_path) for video_path in
                            video_files_test["video"].values]
        video_files_train = [os.path.join(config.ST_MAPS_PATH, video_path) for video_path in
                             video_files_train["video"].values]

        train_loss_data = []
        for idx, video_file_path in enumerate(video_files_train):
            print(f"Training {idx + 1}/{len(video_files_train)} video files in current Fold: {k}")
            # print(f"Reading Current File: {video_file_path}")
            train_set = DataLoaderRhythmNet(data_path=video_file_path, target_signal_path=config.TARGET_SIGNAL_DIR, clip_size=config.CLIP_SIZE)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=None,
                num_workers=config.NUM_WORKERS,
                shuffle=False
            )
            # print('\nTrainLoader constructed successfully!')

            # Code to use multiple GPUs (if available)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)

            # --------------------------------------
            # Load checkpointed model (if  present)
            # --------------------------------------
            model, optimizer, loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path)
            if checkpoint_flag:
                print(f"Checkpoint Found! Loading from checkpoint :: LOSS={loss}")
            else:
                print("Checkpoint Not Found! Training from beginning")

            # -----------------------------
            # Start training
            # -----------------------------

            train_loss_data_per_epoch = []
            # train_loss = 0.0
            for epoch in tqdm(range(config.EPOCHS), leave=True, position=0):
                # training
                train_loss = engine.train_fn(model, train_loader, optimizer, loss_fn, save_model=True)

                # print(f"\n[Epoch: {epoch + 1}/{config.EPOCHS} ",
                #       "Training Loss: {:.3f} ".format(train_loss))

                train_loss_data_per_epoch.append(train_loss)

            mean_loss = np.mean(train_loss_data_per_epoch)
            train_loss_data.append(mean_loss)
            print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")
            save_model_checkpoint(model, optimizer, mean_loss, config.CHECKPOINT_PATH)

        test_loss_data = []
        rmse_hr = []
        for idx, video_file_path in enumerate(video_files_test):
            print(f"Training {idx + 1}/{len(video_files_test)} video files")
            print(f"Reading Current File: {video_file_path}")
            test_set = DataLoaderRhythmNet(data_path=video_file_path, target_signal_path=config.TARGET_SIGNAL_DIR, clip_size=config.CLIP_SIZE)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=None,
                num_workers=config.NUM_WORKERS,
                shuffle=False
            )

            print('\nTestLoader constructed successfully!')

            # Code to use multiple GPUs (if available)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)

            # --------------------------------------
            # Load checkpointed model (if  present)
            # --------------------------------------
            model, optimizer, loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path)

            # -----------------------------
            # Start Validation
            # -----------------------------

            print(f"Starting training for {config.EPOCHS} Epochs")

            test_loss_data_per_epoch = []
            eval_loss = 0.0
            for epoch in tqdm(range(config.EPOCHS), leave=True, position=0):
                # validation
                eval_preds, eval_loss = engine.eval_fn(model, test_loader, loss_fn)

                # print(f"Epoch {epoch} => Val Loss: {eval_loss}")

                test_loss_data_per_epoch.append(eval_loss)

            test_loss_data.append((np.mean(test_loss_data_per_epoch)))

            print(f"Avg Validation Loss: {np.mean(test_loss_data_per_epoch)} for {config.EPOCHS} epochs")
        plot_train_test_curves(train_loss_data, test_loss_data, plot_path=config.PLOT_PATH, fold_tag=k)
        print("done")


if __name__ == '__main__':
    run_training()
