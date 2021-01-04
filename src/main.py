import os
import glob
import torch
import numpy as np
from pprint import pprint
from torchvision import datasets, transforms
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
import engine
import config
from utils.dataset import DataLoaderRhythmNet
from utils.model_utils import plot_loss, load_model_if_checkpointed
from models.simpleCNN import SimpleCNN
from models.lenet import LeNet


def run_training():
    # A simple transform which we will apply to the MNIST images
    # simple_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # check path to checkpoint directory
    if config.CHECKPOINT_PATH:
        if not os.path.exists(config.CHECKPOINT_PATH):
            os.makedirs(config.CHECKPOINT_PATH)
            print("Output directory is created")

    # --------------------------------------
    # Initialize Model
    # --------------------------------------

    model = LeNet()

    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(42)
    else:
        print("GPU not available, using CPU")

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.8, patience=5, verbose=True
    # )
    loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------
    # Build Dataloaders
    # --------------------------------------

    testset = trainset = None
    videos = glob.glob(config.DATA_DIR + '*.avi')
    train_set = DataLoaderRhythmNet(data_path=videos[0], target_signal_path=config.TARGET_SIGNAL_DIR)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    test_set = DataLoaderRhythmNet(data_path=config.DATA_DIR, target_signal_path=config.TARGET_SIGNAL_DIR)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    print('\nDataLoaders constructed successfully!')

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

    print(f"Starting training for {config.EPOCHS} Epochs")

    train_loss_data = []
    test_loss_data = []
    for epoch in range(config.EPOCHS):
        # training
        train_loss = engine.train_fn(model, train_loader, optimizer, loss_fn, save_model=True)

        # validation
        eval_preds, eval_loss = engine.eval_fn(model, test_loader, loss_fn)
        eval_loss = 0.0

        print(f"Epoch {epoch} => Training Loss: {train_loss}, Val Loss: {eval_loss}")

        train_loss_data.append(train_loss)
        test_loss_data.append(eval_loss)

    # print(train_dataset[0])
    plot_loss(train_loss_data, test_loss_data, plot_path=config.PLOT_PATH)
    print("done")


if __name__ == '__main__':
    run_training()
