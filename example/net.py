import torch.nn.functional as F
import torch.nn as nn


class ClsNet(nn.Module):
    def __init__(self, num_neuron):
        super(ClsNet, self).__init__()
        self.fc1 = nn.Linear(30, num_neuron)
        self.fc2 = nn.Linear(num_neuron, 64)
        self.cls = nn.Linear(64, 7)

    def forward(self, x):
        x = x.view(-1, 30)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.cls(x)
        return x
