import torch.nn as nn

"""
model.py

Defines the flexible MLP architecture for the AFL Fantasy breakout prediction task.
This version builds layer sizes, dropouts, and structure directly from the configuration file.

Usage:
    model = BreakoutPredictorMLP(input_dim=num_features, hidden_dims=[256, 128, 64], input_dropout=0.3, output_dropout=0.3)
    output = model(input_tensor)  # output will be raw logits (use BCEWithLogitsLoss)
"""

class BreakoutPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, input_dropout=0.3, output_dropout=0.3):
        super(BreakoutPredictorMLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(input_dropout))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(input_dropout))

        # Output layer (no Sigmoid; BCEWithLogitsLoss expects raw logits)
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)