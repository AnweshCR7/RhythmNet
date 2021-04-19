import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import ssl
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_stdlib_context


def plot_activations(activations, square=8, name="plot", limit=3):
    # square = 8
    ix = 0
    activations = activations.squeeze(0)
    # limit = 2
    fig = plt.figure()
    for _ in range(2):
        for _ in range(2):
            # specify subplot and turn of axis
            # ax = plt.subplot(square, square, ix+1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(activations[ix, :, :].permute(1,0))
            plt.show()
            ix += 1
            if ix == limit:
                break

    # # show the figure
    # plt.show()
    # fig.savefig(f'./{name}.png', dpi=fig.dpi)


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
        self.resnet_linear = nn.Linear(512, 1000)
        # Fully connected layer to regress the o/p of resnet -> 1 HR per clip
        self.fc_regression = nn.Linear(1000, 1)
        self.rnn = nn.GRU(input_size=10, hidden_size=10)
        self.fc = nn.Linear(10, 10)

    def forward(self, st_maps, target):
        batched_output_per_clip = []

        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        st_maps = st_maps.unsqueeze(0)
        for t in range(st_maps.size(1)):
            # with torch.no_grad():
            # x = self.resnet18[0](st_maps[:, t, :, :, :])
            # plot_activations(x)
            x = self.resnet18(st_maps[:, t, :, :, :])
            # collapse dimensions to BSx512 (resnet o/p)
            x = x.view(x.size(0), -1)
            # output dim: BSx1
            x = self.resnet_linear(x)
            x = self.fc_regression(x)
            # normalize by frame-rate
            x = x*25.0
            batched_output_per_clip.append(x.squeeze(0))
            # input should be (seq_len, batch, input_size)

        output_seq = torch.stack(batched_output_per_clip, dim=0).transpose_(0, 1)
        # gru_output, h_n = self.rnn(output_seq.unsqueeze(1))

        # fc_out = self.fc(gru_output.flatten())
        #
        # return output_seq, gru_output.squeeze(0), fc_out
        return output_seq

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
