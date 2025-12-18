from typing import Literal
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from models.model_helpers import IdentityModule

    
ACTIVATIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
    "no_activation": IdentityModule,
}

# MARK: ResNet
class CustomResNet(nn.Module):
    def __init__(self, size:Literal["18", "34", "50", "101", "152"], weights="DEFAULT", in_dim=3, out_dim=1000, flatten_out=False, out_activation:Literal["relu", "lrelu", "sigmoid", "tanh", "no_activation", "heavyside", "softmax"]|None=None) -> None:
        super().__init__()
        assert flatten_out == False or out_dim == 1, "output can only be flattened to [BS] if there is exactly one output channel"

        in_dim = int(in_dim)
        out_dim = int(out_dim)
        out_activation = None if out_activation == "no_activation" else out_activation

        self.flatten_out = flatten_out

        sizes = {
            "18": resnet18,
            "34": resnet34,
            "50": resnet50,
            "101": resnet101,
            "152": resnet152,
        }
        self.model = sizes[size](weights=weights)

        if in_dim != 3:
            self.model.conv1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if out_dim != 1000:
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=out_dim)
            
        if out_activation != None:
            self.model = torch.nn.Sequential(self.model, ACTIVATIONS[out_activation]())

        self.options = {
            "size": size,
            "weights": weights,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "flatten_out": flatten_out,
            "out_activation": out_activation,
        }
        
    def forward(self, x):
        x = self.model(x)
        if self.flatten_out:
            x = torch.flatten(x, start_dim=0)
        return x

# MARK: MLP
class MLP(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, hidden_channels:list[int], activation="relu", final_activation="no_activation", flatten_out=False) -> None:
        '''
        activation and final_activation must be "sigmoid", "tanh", "relu" or "lrelu" (leaky relu with slope 0.01), or "no_activation"
        '''
        super().__init__()
        assert activation in ACTIVATIONS and final_activation in ACTIVATIONS
        self.flatten_out = flatten_out
        self.out_channels = out_channels

        channels = [in_channels] + hidden_channels + [out_channels]

        self.layers = nn.Sequential()
        for i in range(len(channels) - 1):
            self.layers.add_module(f"layer_{i}", self.basic_block(channels[i], channels[i+1]))
            if i < (len(channels) - 2): # not the output layer               
                self.layers.add_module(f"{activation}_{i}", ACTIVATIONS[activation]())
        self.layers.add_module("final_activation", ACTIVATIONS[final_activation]())

        self.options = {
            "out_channels": out_channels,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "activation": activation,
            "final_activation": final_activation,
            "flatten_out": flatten_out,
        }

    def basic_block(self, in_channels, out_channels) -> torch.nn.Module:
        return nn.Linear(in_channels, out_channels)

    def forward(self, x):
        if len(x.shape) == 1:
            # if the inputs are scalars. Linear expects 1 row per sample.
            x = torch.unsqueeze(x, 1)
        elif len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)

        x = self.layers(x)
        
        if self.flatten_out or self.out_channels == 1:
            x = torch.flatten(x)
        
        return x
    