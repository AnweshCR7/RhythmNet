import os
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import engine
import engine_vipl
import config
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import DataLoaderRhythmNet
from utils.plot_scripts import plot_train_test_curves, bland_altman_plot, gt_vs_est, create_plot_for_tensorboard
from utils.model_utils import plot_loss, load_model_if_checkpointed, save_model_checkpoint
from models.simpleCNN import SimpleCNN
from models.lenet import LeNet
from models.rhythmNet import RhythmNet
from loss_func.rhythmnet_loss import RhythmNetLoss
from scipy.stats.stats import pearsonr


# Needed in VIPL dataset where each data item has a different number of frames/maps
def collate_fn(batch):
    batched_st_map, batched_targets = [], []
    # for data in batch:
    #     batched_st_map.append(data["st_maps"])
    #     batched_targets.append(data["target"])
    # # torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
    return batch


def rmse(l1, l2):

    return np.sqrt(np.mean((l1-l2)**2))


def mae(l1, l2):

    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])


def compute_criteria(target_hr_list, predicted_hr_list):
    pearson_per_signal = []
    HR_MAE = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    HR_RMSE = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    # for (gt_signal, predicted_signal) in zip(target_hr_list, predicted_hr_list):
    #     r, p_value = pearsonr(predicted_signal, gt_signal)
    #     pearson_per_signal.append(r)

    # return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "Pearson": np.mean(pearson_per_signal)}
    return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE}


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    # loss_fn = nn.L1Loss()
    loss_fn = RhythmNetLoss()

    testset = trainset = None

    # Initialize SummaryWriter object
    writer = SummaryWriter()

    # Read from a pre-made csv file that contains data divided into folds for cross validation
    folds_df = pd.read_csv(config.SAVE_CSV_PATH)

    # Loop for enumerating through folds.
    print(f"Details: {len(folds_df['iteration'].unique())} fold training for {config.EPOCHS} Epochs (each video)")
    # for k in folds_df['iteration'].unique():
    for k in [1]:
        # Filter DF
        video_files_test = folds_df.loc[(folds_df['iteration'] == k) & (folds_df['set'] == 'V')]
        video_files_train = folds_df.loc[(folds_df['iteration'] == k) & (folds_df['set'] == 'T')]

        # Get paths from filtered DF VIPL
        video_files_test = [os.path.join(config.ST_MAPS_PATH, video_path.split('/')[-1]) for video_path in
                            video_files_test["video"].values]
        video_files_train = [os.path.join(config.ST_MAPS_PATH, video_path.split('/')[-1]) for video_path in
                             video_files_train["video"].values]

        # video_files_test = [os.path.join(config.ST_MAPS_PATH, video_path) for video_path in
        #                     video_files_test["video"].values]
        # video_files_train = [os.path.join(config.ST_MAPS_PATH, video_path) for video_path in
        #                      video_files_train["video"].values]

        # video_files_train = video_files_train[:32]
        # video_files_test = video_files_test[:32]

        # print(f"Reading Current File: {video_files_train[0]}")

        # --------------------------------------
        # Build Dataloaders
        # --------------------------------------

        train_set = DataLoaderRhythmNet(st_maps_path=video_files_train, target_signal_path=config.TARGET_SIGNAL_DIR)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            collate_fn=collate_fn
        )
        print('\nTrain DataLoader constructed successfully!')

        # Code to use multiple GPUs (if available)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        # --------------------------------------
        # Load checkpointed model (if  present)
        # --------------------------------------
        if config.DEVICE == "cpu":
            load_on_cpu = True
        else:
            load_on_cpu = False
        model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path, load_on_cpu=load_on_cpu)
        if checkpoint_flag:
            print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
        else:
            print("Checkpoint Not Found! Training from beginning")

        # -----------------------------
        # Start training
        # -----------------------------

        train_loss_per_epoch = []
        for epoch in range(config.EPOCHS):
            # short-circuit for evaluation
            if k == 1:
                break
            target_hr_list, predicted_hr_list, train_loss = engine_vipl.train_fn(model, train_loader, optimizer, loss_fn)

            # Save model with final train loss (script to save the best weights?)
            if checkpointed_loss != 0.0:
                if train_loss < checkpointed_loss:
                    save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
                    checkpointed_loss = train_loss
                else:
                    pass
            else:
                if len(train_loss_per_epoch) > 0:
                    if train_loss < min(train_loss_per_epoch):
                        save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
                else:
                    save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)

            metrics = compute_criteria(target_hr_list, predicted_hr_list)

            for metric in metrics.keys():
                writer.add_scalar(f"Train/{metric}", metrics[metric], epoch)

            print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
                  "\nTraining Loss: {:.3f} |".format(train_loss),
                  "HR_MAE : {:.3f} |".format(metrics["MAE"]),
                  "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)
                  # "Pearsonr : {:.3f} |".format(metrics["Pearson"]), )

            train_loss_per_epoch.append(train_loss)
            writer.add_scalar("Loss/train", train_loss, epoch+1)

            # Plots on tensorboard
            ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
            gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
            writer.add_image('BA_plot', ba_plot_image, epoch)
            writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)

        mean_loss = np.mean(train_loss_per_epoch)
        # Save the mean_loss value for each video instance to the writer
        print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")
        writer.flush()

        # --------------------------------------
        # Load checkpointed model (if  present)
        # --------------------------------------
        if config.DEVICE == "cpu":
            load_on_cpu = True
        else:
            load_on_cpu = False
        model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer,
                                                                                          checkpoint_path,
                                                                                          load_on_cpu=load_on_cpu)
        if checkpoint_flag:
            print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
        else:
            print("Checkpoint Not Found! Training from beginning")

        # -----------------------------
        # Start Validation
        # -----------------------------
        test_set = DataLoaderRhythmNet(st_maps_path=video_files_test, target_signal_path=config.TARGET_SIGNAL_DIR)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            shuffle=False,
            collate_fn=collate_fn
        )
        print('\nEvaluation DataLoader constructed successfully!')

        print(f"Finished Training, Validating {len(video_files_test)} video files for {config.EPOCHS_TEST} Epochs")

        eval_loss_per_epoch = []
        for epoch in range(config.EPOCHS_TEST):
            # validation
            target_hr_list, predicted_hr_list, test_loss = engine_vipl.eval_fn(model, test_loader, loss_fn)

            # truth_hr_list.append(target)
            # estimated_hr_list.append(predicted)
            metrics = compute_criteria(target_hr_list, predicted_hr_list)
            for metric in metrics.keys():
                writer.add_scalar(f"Test/{metric}", metrics[metric], epoch)

            print(f"\nFinished Test [Epoch: {epoch + 1}/{config.EPOCHS_TEST}]",
                  "\nTest Loss: {:.3f} |".format(test_loss),
                  "HR_MAE : {:.3f} |".format(metrics["MAE"]),
                  "HR_RMSE : {:.3f} |".format(metrics["RMSE"]),)

            writer.add_scalar("Loss/test", test_loss, epoch)

            # Plots on tensorboard
            ba_plot_image = create_plot_for_tensorboard('bland_altman', target_hr_list, predicted_hr_list)
            gtvsest_plot_image = create_plot_for_tensorboard('gt_vs_est', target_hr_list, predicted_hr_list)
            writer.add_image('BA_plot', ba_plot_image, epoch)
            writer.add_image('gtvsest_plot', gtvsest_plot_image, epoch)


        # print(f"Avg Validation Loss: {mean_test_loss} for {config.EPOCHS_TEST} epochs")
        writer.flush()
        # plot_train_test_curves(train_loss_data, test_loss_data, plot_path=config.PLOT_PATH, fold_tag=k)
        # Plots on the local storage.
        gt_vs_est(target_hr_list, predicted_hr_list, plot_path=config.PLOT_PATH)
        bland_altman_plot(target_hr_list, predicted_hr_list, plot_path=config.PLOT_PATH)
        writer.close()
        print("done")


if __name__ == '__main__':
    run_training()
