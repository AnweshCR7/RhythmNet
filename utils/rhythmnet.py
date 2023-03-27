"""Backbone CNN for RhythmNet model is a RestNet-18.

This script defines a custom neural network model, RhythmNet, which uses a ResNet-18 backbone convolutional
neural network (CNN) architecture. The ResNet-18  is used as a feature extractor.
The extracted features are then passed through a linear layer for regression
and a one-layer Gated Recurrent Unit (GRU) for time-series modeling. The final output of the model
is the predicted heart rate per clip, both through regression and time-series modeling.

Attributes:
    resnet18 (nn.Sequential): A sequential neural network module that consists of all the layers in the
    ResNet-18 model up to the last average pooling layer.

    resnet_linear (nn.Linear): A linear layer that maps the output of ResNet-18 to a 1000-dimensional feature space.

    fc_regression (nn.Linear): A linear layer that maps the 1000-dimensional features to a scalar value of
    heart rate through regression.

    gru_fc_out (nn.Linear): A linear layer that maps the output of the GRU to a scalar value of heart rate.

    rnn (nn.GRU): A one-layer GRU module that models the time-series information in the extracted features.

Methods:
    forward(st_maps): Forward pass through the RhythmNet model.

Args:
    st_maps (torch.Tensor): A tensor  containing the input spatial-temporal maps of face regions that
    encode the subtle changes in skin color caused by the heartbeat.
Returns:
    tuple: A tuple of two torch.Tensor objects, containing the predicted heart rate per clip through regression
    and time-series modeling respectively.
"""

import torch
from torch import nn
from torchvision import models


class RhythmNet(nn.Module):
    def __init__(self):
        super().__init__()

        # resnet o/p -> bs x 1000
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        # The resnet average pool layer before fc
        self.resnet_linear = nn.Linear(512, 1000)
        self.fc_regression = nn.Linear(1000, 1)
        self.gru_fc_out = nn.Linear(1000, 1)
        self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=1)

    def forward(self, st_maps, target):
        batched_output_per_clip = []
        gru_input_per_clip = []
        hr_per_clip = []

        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        st_maps = st_maps.unsqueeze(0)

        for batch_num in range(st_maps.size(1)):

            x_features = self.resnet18(st_maps[:, batch_num, :, :, :])
            # collapse dimensions to BSx512 (resnet o/p)
            x_features = x_features.view(x_features.size(0), -1)
            # output dim: BSx1 and Squeeze sequence length after completing GRU step
            x_features = self.resnet_linear(x_features)
            # Save CNN features per clip for the GRU
            gru_input_per_clip.append(x_features.squeeze(0))

            # Final regression layer for CNN features -> HR (per clip)
            x_features = self.fc_regression(x_features)
            # normalize HR by frame-rate: 25.0 for VIPL
            x_features = x_features * 25.0
            batched_output_per_clip.append(x_features.squeeze(0))
            # input should be (seq_len, batch, input_size)

        # the features extracted from the backbone CNN are fed to a one-layer GRU structure.
        regression_output = torch.stack(batched_output_per_clip, dim=0).permute(1, 0)

        # Trying out GRU in addition to the regression now.
        gru_input = torch.stack(gru_input_per_clip, dim=0)
        gru_output, _ = self.rnn(gru_input.unsqueeze(1))
        # gru_output = gru_output.squeeze(1)
        for i in range(gru_output.size(0)):
            heart_rate = self.gru_fc_out(gru_output[i, :, :])
            hr_per_clip.append(heart_rate.flatten())

        gru_output_seq = torch.stack(hr_per_clip, dim=0).permute(1, 0)
        # return output_seq, gru_output.squeeze(0), fc_out
        return regression_output, gru_output_seq.squeeze(0)[:6]

    def name(self):
        return "RhythmNet"
