import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegression(nn.Module):
    def __init__(self, input_feat):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_feat, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.linear(x))
    
    def get_model(input_feat):
        model = LogisticRegression(input_feat)
        return model