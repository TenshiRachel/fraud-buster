from src.process_data import get_train_data
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=True)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train.to_numpy())
y_train_tensor = torch.LongTensor(y_train.to_numpy())

X_test_tensor = torch.FloatTensor(X_test.to_numpy())
y_test_tensor = torch.LongTensor(y_test.to_numpy())

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoader for batch processing
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(60, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
input_size = X_train.shape[1]  # Dynamic input size
model = MLP(input_size).to(device)

# Set up optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # Debug print to check the shape of the labels
        # print(f"Labels shape before reshaping: {labels.shape}")  # Debug print for label shape
        
        # Make sure the labels are the correct shape (flatten if needed)
        if labels.dim() > 1:
            labels = labels.view(-1)  # Flatten the tensor to 1D if necessary

        # Debug print to check the shape of the labels after reshaping
        # print(f"Labels shape after reshaping: {labels.shape}")  # Debug print after reshaping

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}')


# evaluate model
model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        # Store for F1-score calculation
        all_predictions.extend(predictions.cpu().numpy())  # Convert to list
        all_labels.extend(labels.cpu().numpy())  # Convert to list

# accuracy calc 
accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Compute F1-score
f1 = f1_score(all_labels, all_predictions, average='weighted')  # Use 'macro' or 'weighted' depending on needs
print(f'Test F1 Score: {f1:.4f}')