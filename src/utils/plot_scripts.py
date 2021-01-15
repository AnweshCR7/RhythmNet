import os
import numpy as np
# from utils.read_data import plot_signal
import matplotlib.pyplot as plt


def plot_train_test_curves(train_loss_data, test_loss_data, plot_path, fold_tag=1):
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    clip = min(len(train_loss_data), len(test_loss_data))
    x_ax = np.arange(1, clip+1)
    fig = plt.figure()
    plt.plot(x_ax, train_loss_data[:clip], label="train_loss")
    plt.plot(x_ax, test_loss_data[:clip], label="test_loss")
    plt.title('Train-Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Num Epoch')
    plt.legend(loc='best')
    plt.show()
    fig.savefig(plot_path+f'/loss_fold_{fold_tag}.png', dpi=fig.dpi)


# def plot_test_output(test_op, plot_path):
#     if not os.path.exists(plot_path):
#         os.makedirs(plot_path)
#
#     x_ax = np.arange(1, len(test_op)+1)
#     fig = plt.figure()
#     plt.plot(x_ax, test_op, label="predicted_HR")
#     plt.ylabel('HR')
#     plt.xlabel('Time')
#     plt.show()
#     fig.savefig(plot_path + f'/Test_output.png', dpi=fig.dpi)
#
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
    plot_signal('data/data_preprocessed', 's22_trial05')