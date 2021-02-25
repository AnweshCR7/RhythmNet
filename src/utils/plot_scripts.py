import os
import io
import numpy as np
import PIL.Image
from torchvision.transforms import ToTensor
import config as config
# from utils.read_data import plot_signal
import matplotlib.pyplot as plt


def plot_train_test_curves(train_loss_data, test_loss_data, plot_path, fold_tag=1):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    clip = min(len(train_loss_data), len(test_loss_data))
    x_ax = np.arange(1, clip + 1)
    fig = plt.figure()
    plt.plot(x_ax, train_loss_data[:clip], label="train_loss")
    plt.plot(x_ax, test_loss_data[:clip], label="test_loss")
    plt.title('Train-Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Num Epoch')
    plt.legend(loc='best')
    plt.show()
    fig.savefig(plot_path + f'/loss_fold_{fold_tag}.png', dpi=fig.dpi)


def gt_vs_est(data1, data2, plot_path=None, tb=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    # mean = np.mean([data1, data2], axis=0)
    # diff = data1 - data2                   # Difference between data1 and data2
    # md = np.mean(diff)                   # Mean of the difference
    # sd = np.std(diff, axis=0)            # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(data1, data2)
    plt.title('true labels vs estimated')
    plt.ylabel('estimated HR')
    plt.xlabel('true HR')
    # plt.axhline(md,           color='gray', linestyle='--')
    # plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    # plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    else:
        # plt.show()
        fig.savefig(plot_path + f'/true_vs_est.png', dpi=fig.dpi)


def bland_altman_plot(data1, data2, plot_path=None, tb=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    fig = plt.figure()
    plt.scatter(mean, diff)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    if tb:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    else:
        # plt.show()
        fig.savefig(plot_path + f'/bland-altman_new.png', dpi=fig.dpi)


def create_plot_for_tensorboard(plot_name, data1, data2):
    if plot_name == "bland_altman":
        fig_buf = bland_altman_plot(data1, data2, tb=True)

    if plot_name == "gt_vs_est":
        fig_buf = gt_vs_est(data1, data2, tb=True)

    image = PIL.Image.open(fig_buf)
    image = ToTensor()(image)

    return image

#
# def plot_rmse(data, plot_path, fold=0):
#     if not os.path.exists(plot_path):
#         os.makedirs(plot_path)
#
#     x_ax = np.arange(1, len(data)+1)
#     fig = plt.figure()
#     plt.plot(x_ax, data, label="predicted_HR_RMSE")
#     plt.ylabel('RMSE_HR')
#     plt.xlabel('Time')
#     plt.show()
#     fig.savefig(plot_path + f'/RMSE_fold{fold}.png', dpi=fig.dpi)


if __name__ == '__main__':
    # plot_signal('data/data_preprocessed', 's22_trial05')
    gt_vs_est(np.random.random(100), np.random.random(100), plot_path=config.PLOT_PATH)
