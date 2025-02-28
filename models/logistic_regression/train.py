import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.process_data import get_train_data
from models.logistic_regression.model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_logistic_regression(feature_engineering=True, n_iterations=400, batch_size=512):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Load processed dataset
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_engineering)

    #Convert to PyTorch Tensors
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32)).to(device)
    X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.to_numpy().astype(np.float32)).to(device)

    #Reshape target tensors for BCELoss compatibility
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)

    #Create DataLoader for batch training
    batch_size = 1024
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Initialize the model
    model = LogisticRegression(input_feat = X_train.shape[1]).to(device)

    #Define loss function
    loss = nn.BCELoss()
    optimiser = torch.optim.SGD(params = model.parameters(), lr=0.01)

    #Training loop
    n_iterations = 400

    for epoch in range(n_iterations):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            #forward pass
            y_preds = model(batch_X)
            L = loss(y_preds, batch_y)

            #Backprop
            L.backward()

            #Update parameters
            optimiser.step()

            #Zero gradients
            optimiser.zero_grad()

        #Print loss
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss {L.item():.3f}')

    #Evaluate model
    with torch.no_grad():
        accuracy = model(X_test).round().eq(y_test).sum().item() / len(y_test)
        print(f'Test Accuracy: {100*accuracy:.2f} %')