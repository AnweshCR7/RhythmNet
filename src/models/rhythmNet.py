import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from models.resnet import resnet18
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

'''
Backbone CNN for RhythmNet model is a RestNet-18
'''


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()

        # resnet o/p -> bs x 1000
        self.resnet18 = resnet18(pretrained=False)
        self.rnn = nn.GRU(input_size=1000, hidden_size=1)

    def forward(self, frame, target):
        x = self.resnet18(frame)

        output, h_n = self.rnn(x.unsqueeze(1))

        return torch.mean(output, dim=0)

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
