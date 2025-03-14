import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),  # Added BatchNorm
            nn.LeakyReLU(0.1),  # Changed to LeakyReLU
            nn.Dropout(0.35),  # Added Dropout to prevent overfitting
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  
            nn.LeakyReLU(0.1),
            nn.Dropout(0.35),  
            nn.Linear(32, 2)  # Output layer for binary classification
        )

    def forward(self, x):
        return self.model(x)

def create_model(input_size, class_weights, device):
    model = MLP(input_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    return model, criterion, optimizer
