# run this file to train neural_network and get performance metrics 

from src.process_data import get_train_data
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score, roc_auc_score, average_precision_score

# Load and preprocess data
X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=False)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train.to_numpy())
X_test_tensor = torch.FloatTensor(X_test.to_numpy())

y_train_tensor = torch.LongTensor(y_train.to_numpy().reshape(-1))  
y_test_tensor = torch.LongTensor(y_test.to_numpy().reshape(-1))  

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 128  # Increased batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define improved MLP model
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

# Initialize model
input_size = X_train.shape[1]
model = MLP(input_size).to(device)

# Adjusted loss function (handle class imbalance)
class_weights = torch.tensor([1.0, 4.5]).to(device)  # Fraud class is 10x more important
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Use Adam optimizer for better convergence
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Training loop
num_epochs = 35 # Increased epochs for better learning
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.view(-1)  # Ensure it's 1D
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')

# Evaluate model
model.eval()
all_predictions = []
all_labels = []
all_probs = []  # Store probabilities for AUC-ROC

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for fraud class (1)
        probabilities = outputs[:, 1].sigmoid()  # Use sigmoid to get probability values
        predictions = (probabilities > 0.6).long()  # Adjusted threshold for better recall
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

# Compute metrics
accuracy = sum([p == l for p, l in zip(all_predictions, all_labels)]) / len(all_labels)
f1 = f1_score(all_labels, all_predictions, average='weighted')
balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
auc_roc = roc_auc_score(all_labels, all_probs)  # New metric to evaluate fraud detection
roc_pr = average_precision_score(all_labels, all_probs)  # Added Precision-Recall AUC Score

# Display results
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test F1 Score: {f1:.4f}')
print(f'Test Balanced Accuracy: {balanced_acc:.4f}')
print(f'Test AUC-ROC: {auc_roc:.4f}')  # AUC-ROC score for fraud detection
print(f'Test ROC-PR Score: {roc_pr:.4f}')  # ✅ Print ROC-PR Score

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, digits=4))
