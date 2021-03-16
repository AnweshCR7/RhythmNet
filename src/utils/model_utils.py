import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss_data, test_loss_data, plot_path):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    x_ax = np.arange(1, len(train_loss_data)+1)
    fig = plt.figure()
    plt.plot(x_ax, train_loss_data, label="train_loss")
    plt.plot(x_ax, test_loss_data, label="test_loss")
    plt.title('Train-Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Num Epoch')
    plt.legend(loc='best')
    plt.show()
    fig.savefig(plot_path+'/train-test_loss.png', dpi=fig.dpi)


def save_model_checkpoint(model, optimizer, loss, checkpoint_path):
    save_filename = "running_model.pt"
    # checkpoint_path = os.path.join(checkpoint_path, save_filename)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save({
        # 'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(checkpoint_path, save_filename))
    print('Saved!')


def load_model_if_checkpointed(model, optimizer, checkpoint_path, load_on_cpu=False):
    loss = 0.0
    checkpoint_flag = False

    # check if checkpoint exists
    if os.path.exists(os.path.join(checkpoint_path, "running_model.pt")):
        checkpoint_flag = True
        if load_on_cpu:
            checkpoint = torch.load(os.path.join(checkpoint_path, "running_model.pt"), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(checkpoint_path, "running_model.pt"))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    return model, optimizer, loss, checkpoint_flag
