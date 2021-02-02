import torch.nn as nn
import torch


class RhythmNetLoss(nn.Module):
    def __init__(self, weight=100.0):
        super(RhythmNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambd = weight

    def forward(self, outputs, target):
        resnet_outputs, gru_outputs = outputs
        target_array = target.repeat(1, resnet_outputs.shape[1])
        l1_loss = self.l1_loss(resnet_outputs, target_array)
        # smooth_loss = self.smooth_loss(gru_outputs)
        loss = l1_loss + self.lambd*self.smooth_loss(gru_outputs)
        return loss

    # Need to write backward pass for this los function
    def smooth_loss(self, gru_outputs):
        gru_outputs = gru_outputs.flatten()
        hr_mean = gru_outputs[:6].mean()
        smooth_loss = sum([abs(gru_output - hr_mean) for gru_output in gru_outputs[:6]])
        return smooth_loss/6
        # l_smooth =
    # def backward(self):
