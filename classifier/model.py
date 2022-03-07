import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, sbert_dim, n_classes, hidden_neurons_1=256, hidden_neurons_2=128, dropout_rate=0.5):
        super().__init__()

        self.fc1 = nn.Linear(sbert_dim, hidden_neurons_1)
        self.fc2 = nn.Linear(hidden_neurons_1, hidden_neurons_2)
        self.fc3 = nn.Linear(hidden_neurons_2, n_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_activation=F.relu):
        x = hidden_activation(self.fc1(x))
        x = self.dropout(x)
        x = hidden_activation(self.fc2(x))
        x = self.dropout(x)
        return F.softmax(self.fc3(x))
