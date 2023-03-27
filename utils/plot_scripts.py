import os
import io

import PIL
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor


def gt_vs_est(data1, data2, plot_path=None, to_buffer=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    fig, axis = plt.subplots()
    axis.scatter(data1, data2)
    axis.set_title('true labels vs estimated')
    axis.set_ylabel('estimated HR')
    axis.set_xlabel('true HR')

    if to_buffer:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf

    fig.savefig(os.path.join(plot_path, 'true_vs_est.png'), dpi=fig.dpi)
    return None


def bland_altman_plot(data1, data2, plot_path=None, to_buffer=False):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    mean_diff = np.mean(diff)
    standard_deviation = np.std(diff, axis=0)

    fig, axis = plt.subplots()
    axis.scatter(mean, diff)
    axis.axhline(mean_diff, color='gray', linestyle='--')
    axis.axhline(mean_diff + 1.96 * standard_deviation, color='gray', linestyle='--')
    axis.axhline(mean_diff - 1.96 * standard_deviation, color='gray', linestyle='--')
    axis.set_title('Bland-Altman Plot')
    axis.set_ylabel('Difference')
    axis.set_xlabel('Mean')

    if to_buffer:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf

    return fig.savefig(os.path.join(plot_path, 'bland-altman_new.png'), dpi=fig.dpi)


def create_tensorboard_plot(plot_name: str, data1, data2) -> ToTensor:
    """Create a plot for Tensorboard.

    Args:
        plot_name (str): The type of plot to create. Valid options are "bland_altman" and "gt_vs_est".
        data1: The first dataset to plot.
        data2: The second dataset to plot.

    Returns:
        ToTensor: The plot image as a PyTorch tensor.
    """
    plot_funcs = {
        "bland_altman": bland_altman_plot,
        "gt_vs_est": gt_vs_est,
    }

    plot_func = plot_funcs.get(plot_name)
    if plot_func is None:
        raise ValueError(f"Invalid plot name: {plot_name}")

    fig_buf = plot_func(data1, data2, to_buffer=True)
    image = ToTensor()(PIL.Image.open(fig_buf))
    return image
