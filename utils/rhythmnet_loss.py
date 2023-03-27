from torch import nn
import numpy as np
import torch
from torch.autograd import Function

from vital_signs.utils import config
class CustomLoss(Function): # pylint: disable=W0223

    """A custom autograd function for the smooth loss component of the RhythmNet loss function."""

    @staticmethod
    def forward(ctx, hr_t, hr_outs, T):  # pylint: disable=W0221
        """Computes the forward pass of the custom autograd function.

        Args:
            ctx (torch.autograd.function.Context): A context object that can be used to stash
            information for backward computation.

            hr_t (torch.Tensor): A tensor of shape (1,), representing the true heart rate
            for a particular time step.

            hr_outs (torch.Tensor)Args: A tensor of shape (seq_len,), representing the predicted heart
            rates for all time steps.

            T (int): An integer representing the number of time steps.

        Returns:
            torch.Tensor: A tensor of shape (1,), representing the smooth loss for a particular time step."""
        ctx.hr_outs = hr_outs
        ctx.hr_mean = hr_outs.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)

        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t

        return loss

    @staticmethod
    def backward(ctx, grad_output):# pylint: disable=W0221,W0613:
        """Computes the backward pass of the custom autograd function.

        Args:
            ctx (torch.autograd.function.Context): A context object that canbe used to
            stash information for backward computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the gradients
            of the loss with respect to hr_t, hr_outs, and T."""
        output = torch.zeros(1).to(config.DEVICE)

        hr_t, = ctx.saved_tensors
        hr_outs = ctx.hr_outs

        for heart_rate in hr_outs:
            if heart_rate == hr_t:
                pass
            else:
                output = output + (1 / ctx.T) * torch.sign(ctx.hr_mean - heart_rate)

        output = (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None



class RhythmNetLoss(nn.Module):
    """ RhythmNetLoss calculates the loss function for the RhythmNet model.

    It uses a combination of L1 loss and a custom smoothness loss to calculate the final loss.

    Args:
        weight (float): A weight factor to control the relative contribution of the smoothness loss to the final loss.
            Default value is 100.0.

    Attributes:
        l1_loss (nn.L1Loss): L1 loss function provided by PyTorch.
        lambd (float): Weight factor for the smoothness loss.
        gru_outputs_considered (None or Tensor): A tensor of shape (batch_size, seq_len) containing the output of the
            GRU layer of the RhythmNet model, used for computing the smoothness loss. Initialized to None.
            custom_loss (CustomLoss): A custom smoothness loss function that penalizes large changes
            between consecutive output values.
        device (str): The device on which to perform calculations. Default value is 'cpu'.

    Methods:
        forward(resnet_outputs, gru_outputs, target): Calculates the combined loss using L1 loss and smoothness loss.
        smooth_loss(gru_outputs): Calculates the smoothness loss component of the combined loss."""

    def __init__(self, weight=100.0):
        """ Initializes a new instance of the RhythmNetLoss class.

        Args:
            weight (float): A weight factor to control the relative contribution of the smoothness loss to the final
                loss. Default value is 100.0."""

        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight
        self.gru_outputs_considered = None
        self.custom_loss = CustomLoss()
        self.device = 'cpu'

    def forward(self, resnet_outputs, gru_outputs, target):
        """Calculates the combined loss using L1 loss and smoothness loss.

        Args:
            resnet_outputs (Tensor): A tensor of shape (batch_size, num_classes) containing
            the output of the ResNet layer of the RhythmNet model.

            gru_outputs (Tensor): A tensor of shape (batch_size, seq_len, hidden_size)
            containing the output of the GRU layer of the RhythmNet model.

            target (Tensor): A tensor of shape (batch_size, num_classes) containing the target values.

        Returns:
            loss (Tensor): A scalar tensor representing the combined loss."""
        l1_loss = self.l1_loss(resnet_outputs, target)
        smooth_loss_component = self.smooth_loss(gru_outputs)

        loss = l1_loss + self.lambd * smooth_loss_component
        return loss

    def smooth_loss(self, gru_outputs):
        """Calculates the smoothness loss component of the combined loss.

        Args:
            gru_outputs (Tensor): A tensor of shape (batch_size, seq_len, hidden_size)
            containing the output of the GRU layer of the RhythmNet model.

        Returns:
            smooth_loss (Tensor): A scalar tensor representing the smoothness loss."""

        smooth_loss = torch.zeros(1).to(device=self.device)
        self.gru_outputs_considered = gru_outputs.flatten()

        for hr_t in self.gru_outputs_considered:
            smooth_loss = smooth_loss + self.custom_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                               self.gru_outputs_considered,
                                                               self.gru_outputs_considered.shape[0])
        return smooth_loss / self.gru_outputs_considered.shape[0]


def rmse(array_1, array_2):
    """Computes the root mean squared error (RMSE) between two arrays.

    Returns:
    float: RMSE between array_1 and array_2."""

    return np.sqrt(np.mean((array_1 - array_2) ** 2))


def mae(array_1, array_2):
    """Computes the mean absolute error (MAE) between two arrays.

    Returns:
    float: MAE between l1 and l2."""

    return np.mean([abs(item1 - item2) for item1, item2 in zip(array_1, array_2)])


def compute_criteria(target_hr_list, predicted_hr_list):
    """Computes the mean absolute error (MAE) and root mean squared error
    (RMSE) between predicted and target heart rate lists.

    Args:
    target_hr_list (array-like): Target heart rate list.
    predicted_hr_list (array-like): Predicted heart rate list.

    Returns:
    dict: Dictionary containing MAE and RMSE values."""
    hr_mae = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    hr_rmse = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    return {"MAE": np.mean(hr_mae), "RMSE": hr_rmse}
