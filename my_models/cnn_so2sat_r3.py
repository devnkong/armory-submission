import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from art.classifiers import PyTorchClassifier

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def so2sat_model_cnn_2layer(in_ch, in_dim, width, linear_size=128):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4),linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 17)
    )
    return model

def get_art_model(model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None):
    model = so2sat_model_cnn_2layer(in_ch=14, in_dim=32, width=1, linear_size=256)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    wrapped_model = PyTorchClassifier(
        model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
        input_shape=(32, 32, 14),
        nb_classes=17,
        # clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )

    return wrapped_model