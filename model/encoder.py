"""Convnet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Segnet network."""

    def __init__(self, input_nbr, label_nbr, momentum):
        """Init fields."""
        super(ConvNet, self).__init__()

        self.input_nbr = input_nbr

        batchNorm_momentum = momentum

        self.conv1 = nn.Conv2d(input_nbr, input_nbr, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(input_nbr, momentum=batchNorm_momentum)
        self.conv2 = nn.Conv2d(input_nbr, label_nbr, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(label_nbr, momentum=batchNorm_momentum)
        self.conv3 = nn.Conv2d(label_nbr, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward method."""
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.conv3(x2)

        return x3

    def load_from_filename(self, model_path):
        """Load weights method."""
        print("load weights from ConvNet.pth")
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)
