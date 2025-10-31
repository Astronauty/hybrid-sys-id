import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class ModeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_classes):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) # logits

