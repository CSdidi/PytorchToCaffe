import torch.nn.functional as F
import torch.nn as nn


class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.fc1 = nn.Linear(27, 256)
        self.fc2 = nn.Linear(256, 64)
        self.cls = nn.Linear(64, 7)

    def forward(self, x):
        x = x.view(-1, 27)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.cls(x)
        return x
