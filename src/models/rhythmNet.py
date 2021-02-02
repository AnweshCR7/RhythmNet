import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()

        # resnet o/p -> bs x 1000
        # self.resnet18 = resnet18(pretrained=False)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]

        self.resnet18 = nn.Sequential(*modules)
        # The resnet average pool layer before fc
        # self.avgpool = nn.AvgPool2d((10, 1))
        self.fc_resnet = nn.Linear(512, 1)

        self.rnn = nn.GRU(input_size=10, hidden_size=10)
        self.fc = nn.Linear(100, 1)

    def forward(self, st_maps, target):
        output_per_clip = []
        # so as to reflect a batch_size = 1
        st_maps = st_maps.unsqueeze(0)
        for t in range(st_maps.size(1)):
            with torch.no_grad():
                x = self.resnet18(st_maps[:, t, :, :, :])
                # collapse dimensions to BSx512 (resnet o/p)
                x = x.view(x.size(0), -1)
            # output dim: BSx1
            x = self.fc_resnet(x)
            # For now since we're working with BS = 1, lets collapse that dimension
            output_per_clip.append(x.squeeze(0))
            # input should be (seq_len, batch, input_size)

        output_seq = torch.stack(output_per_clip, dim=0).transpose_(0, 1)
        gru_output, h_n = self.rnn(output_seq.unsqueeze(1))
        return output_seq, gru_output.squeeze(0)

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
