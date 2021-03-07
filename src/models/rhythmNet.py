import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
import config

ssl._create_default_https_context = ssl._create_stdlib_context

'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        # The resnet average pool layer before fc
        # self.avgpool = nn.AvgPool2d((10, 1))
        self.fc_resnet = nn.Linear(512, 1)

        self.rnn = nn.GRU(input_size=config.GRU_TEMPORAL_WINDOW, hidden_size=config.GRU_TEMPORAL_WINDOW)
        # self.fc = nn.Linear(config.GRU_TEMPORAL_WINDOW, config.GRU_TEMPORAL_WINDOW)

    def forward(self, st_maps, target):
        batched_output_per_clip = []

        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        st_maps = st_maps.unsqueeze(0)
        for t in range(st_maps.size(1)):
            # with torch.no_grad():
            x = self.resnet18(st_maps[:, t, :, :, :])
            # collapse dimensions to BSx512 (resnet o/p)
            x = x.view(x.size(0), -1)
            # output dim: BSx1
            x = self.fc_resnet(x)
            # normalize by frame-rate: 25.0 for VIPL
            x = x*25.0
            batched_output_per_clip.append(x.squeeze(0))
            # input should be (seq_len, batch, input_size)

        output_seq = torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
        gru_output, h_n = self.rnn(output_seq.unsqueeze(1)[:, :, :config.GRU_TEMPORAL_WINDOW])

        # fc_out = self.fc(gru_output.flatten())
        #
        # return output_seq, gru_output.squeeze(0), fc_out
        return output_seq, gru_output

    def name(self):
        return "RhythmNet"


if __name__ == '__main__':
    # cm = RhythmNet()
    # img = torch.rand(3, 28, 28)
    # target = torch.randint(1, 20, (5, 5))
    # x = cm(img)
    # print(x)
    resnet18 = models.resnet18(pretrained=False)
    print(resnet18)
