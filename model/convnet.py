"""Convnet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Segnet network."""

    def __init__(self, input_nbr, label_nbr):
        """Init fields."""
        super(ConvNet, self).__init__()

        self.input_nbr = input_nbr

        batchNorm_momentum = 0.1

        self.conv1 = nn.Conv2d(input_nbr, input_nbr, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(input_nbr, momentum=batchNorm_momentum)
        self.conv2 = nn.Conv2d(input_nbr, label_nbr, kernel_size=1)

    def forward(self, x):
        """Forward method."""
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.conv2(x1)

        return x2

    def load_from_filename(self, model_path):
        """Load weights method."""
        print("load weights from ConvNet.pth")
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)
