import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def name(self):
        return "SimpleCNN"


if __name__ == '__main__':
    cm = SimpleCNN()
    img = torch.rand(3, 28, 28)
    target = torch.randint(1, 20, (5, 5))
    x = cm(img)
    print(x)
