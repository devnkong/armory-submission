"""
CNN model for 48x48x3 image classification
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from art.classifiers import PyTorchClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


# class Net(nn.Module):
#     """
#     This is a simple CNN for GTSRB and does not achieve SotA performance
#     """
#
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 4, 5, 1)
#         self.conv2 = nn.Conv2d(4, 10, 5, 1)
#         self.fc1 = nn.Linear(810, 500)
#         self.fc2 = nn.Linear(500, 43)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 29, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv3 = nn.Conv2d(29, 59, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(3, stride=2 , ceil_mode=True)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1184, 300)
        self.fc2 = nn.Linear(300, nclasses)
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2_bn = nn.BatchNorm2d(29)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.dense1_bn = nn.BatchNorm1d(300)

    def forward(self, x):
        x =  F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3( self.maxpool2(x))))
        x = F.relu(self.conv4_bn(self.conv4( self.maxpool3(x))))
        x = self.maxpool4(x)
        x = x.view(-1, 1184)
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x)


def make_gtsrb_model(**kwargs) -> Net:
    return Net()

def cross_entropy(outputs_x, targets_x) :
    return -torch.mean(torch.sum(outputs_x * targets_x, dim=1))

def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchClassifier:
    model = make_gtsrb_model(**model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        # loss=nn.CrossEntropyLoss(),
        loss=cross_entropy,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(48, 48, 3),
        nb_classes=43,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model