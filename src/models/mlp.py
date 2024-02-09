import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    基本的なMLPモデル
    """
    def __init__(self, feature_names):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(len(feature_names), 5)
        self.fc2 = nn.Linear(5, 3)
        self.f3 = nn.Linear(3, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.softmax(self.f3(x))


