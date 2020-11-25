from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, hidden_size, proj_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, proj_size)
        )

    def forward(self, x):
        return self.net(x)
