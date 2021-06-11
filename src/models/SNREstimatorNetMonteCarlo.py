import torch.nn as nn
import torch.nn.functional as F
import torch
import random


class SNREstimatorNetMonteCarlo(nn.Module):
    def __init__(self):
        super(SNREstimatorNetMonteCarlo, self).__init__()
        self.is_active_layer = []

    def setup(self, active_layers, max_pool_kernel_size, conv_kernel_size, conv_filter_size):
        self.is_active_layer = active_layers

        max_pool_kernel_size = int(max_pool_kernel_size)
        conv_kernel_size = int(conv_kernel_size)

        conv_init_mean = 0
        conv_init_std = .1
        xavier_normal_gain = 1

        self.bn_input = nn.BatchNorm1d(1)
        nn.init.normal_(self.bn_input.weight, conv_init_mean, conv_init_std)

        output_count = int(conv_filter_size)
        input_count = 1
        self.conv_00 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_00.weight, gain=xavier_normal_gain)
        self.bn_00 = nn.BatchNorm1d(output_count)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[1] == 1:
            input_count = output_count
            self.conv_01 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
            self.bn_01 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[2] == 1:
            input_count = output_count
            self.conv_02 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_02.weight, gain=xavier_normal_gain)
            self.max_pool1d_02 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_02 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_02.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[3] == 1:
            input_count = output_count
            self.conv_10 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_10.weight, gain=xavier_normal_gain)
            self.bn_10 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_10.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[4] == 1:
            input_count = output_count
            self.conv_11 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_11.weight, gain=xavier_normal_gain)
            self.bn_11 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_11.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[5] == 1:
            input_count = output_count
            self.conv_12 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_12.weight, gain=xavier_normal_gain)
            self.max_pool1d_12 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_12 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_12.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[6] == 1:
            input_count = output_count
            self.conv_20 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_20.weight, gain=xavier_normal_gain)
            self.bn_20 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_20.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[7] == 1:
            input_count = output_count
            self.conv_21 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_21.weight, gain=xavier_normal_gain)
            self.bn_21 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_21.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[8] == 1:
            input_count = output_count
            self.conv_22 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_22.weight, gain=xavier_normal_gain)
            self.max_pool1d_22 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_22 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_22.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[9] == 1:
            input_count = output_count
            self.conv_30 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_30.weight, gain=xavier_normal_gain)
            self.bn_30 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_30.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[10] == 1:
            input_count = output_count
            self.conv_31 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_31.weight, gain=xavier_normal_gain)
            self.bn_31 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_31.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[11] == 1:
            input_count = output_count
            self.conv_32 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_32.weight, gain=xavier_normal_gain)
            self.max_pool1d_32 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_32 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_32.weight, conv_init_mean, conv_init_std)

        input_count = output_count
        self.conv_last = nn.Conv1d(input_count, 1, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_last.weight, gain=xavier_normal_gain)

        self.ada_avg_pool1d = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        nonlin = F.elu

        # Some kind of normalization?
        x = x - torch.mean(x, dim=-1).unsqueeze(dim=-1)

        x = self.bn_input(x)

        x = nonlin(self.bn_00(self.conv_00(F.dropout(x, p=0.1, training=self.training))))
        if self.is_active_layer[1] == 1:
            x = nonlin(self.bn_01(self.conv_01(F.dropout(x, p=0.1, training=self.training))))
        if self.is_active_layer[2] == 1:
            x = nonlin(self.bn_02(self.max_pool1d_02(self.conv_02(F.dropout(x, p=0.1, training=self.training)))))
        if self.is_active_layer[3] == 1:
            x = nonlin(self.bn_10(self.conv_10(F.dropout(x, p=0.15, training=self.training))))
        if self.is_active_layer[4] == 1:
            x = nonlin(self.bn_11(self.conv_11(F.dropout(x, p=0.15, training=self.training))))
        if self.is_active_layer[5] == 1:
            x = nonlin(self.bn_12(self.max_pool1d_12(self.conv_12(F.dropout(x, p=0.15, training=self.training)))))
        if self.is_active_layer[6] == 1:
            x = nonlin(self.bn_20(self.conv_20(F.dropout(x, p=0.2, training=self.training))))
        if self.is_active_layer[7] == 1:
            x = nonlin(self.bn_21(self.conv_21(F.dropout(x, p=0.2, training=self.training))))
        if self.is_active_layer[8] == 1:
            x = nonlin(self.bn_22(self.max_pool1d_22(self.conv_22(F.dropout(x, p=0.2, training=self.training)))))
        if self.is_active_layer[9] == 1:
            x = nonlin(self.bn_30(self.conv_30(F.dropout(x, p=0.3, training=self.training))))
        if self.is_active_layer[10] == 1:
            x = nonlin(self.bn_31(self.conv_31(F.dropout(x, p=0.3, training=self.training))))
        if self.is_active_layer[11] == 1:
            x = nonlin(self.bn_32(self.max_pool1d_32(self.conv_32(F.dropout(x, p=0.3, training=self.training)))))

        x = self.conv_last(F.dropout(x, p=0.5, training=self.training))

        x = self.ada_avg_pool1d(x)

        if sum(x.size()[1:]) > x.dim() - 1:
            print(x.size())
            raise ValueError('Check your network idiot!')

        return x
