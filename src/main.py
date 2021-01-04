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
# import dataset
from utils.model_utils import plot_loss
from models.simpleCNN import SimpleCNN
from models.lenet import LeNet


def run_training():
    # A simple transform which we will apply to the MNIST images
    simple_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_set = datasets.MNIST(root=config.DATA_DIR, train=True, transform=simple_transform, download=True)
    test_set = datasets.MNIST(root=config.DATA_DIR, train=False, transform=simple_transform, download=True)

    # train_dataset = dataset.ClassificationDataset(image_paths=train_imgs, targets=train_targets,
    #                                               resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    # test_dataset = dataset.ClassificationDataset(image_paths=test_imgs, targets=test_targets,
    #                                              resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )

    model = LeNet()

    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(42)

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.8, patience=5, verbose=True
    # )
    criterion = nn.CrossEntropyLoss()

    train_loss_data = []
    test_loss_data = []
    for epoch in range(config.EPOCHS):
        # training
        train_loss = engine.train_fn(model, train_loader, optimizer, criterion, save_model=True)

        # validation
        eval_preds, eval_loss = engine.eval_fn(model, test_loader, criterion)

        print(f"Epoch {epoch} => Training Loss: {train_loss}, Val Loss: {eval_loss}")

        train_loss_data.append(train_loss)
        test_loss_data.append(eval_loss)

    # print(train_dataset[0])
    plot_loss(train_loss_data, test_loss_data, plot_path=config.PLOT_PATH)
    print("done")


if __name__ == '__main__':
    run_training()
