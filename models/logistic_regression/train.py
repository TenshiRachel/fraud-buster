import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.process_data import get_train_data
from models.logistic_regression.model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

def plot_roc_pr(y_test, y_prob):
    """ Function to plot ROC and Precision-Recall curves."""

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    roc_pr = auc(recall, precision)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    """ PR Curve """
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.', label=f'PR AUC: {roc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    """ ROC Curve """
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, marker='.', label=f'ROC AUC: {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.show()



def train_logistic_regression(feature_engineering=False, n_iterations=100, batch_size=512):

    #Load processed dataset
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_engineering)

    #Convert to PyTorch Tensors
    X_train = torch.from_numpy(X_train.values.astype(np.float32))
    y_train = torch.from_numpy(y_train.values.astype(np.float32))
    X_test = torch.from_numpy(X_test.values.astype(np.float32))
    y_test = torch.from_numpy(y_test.values.astype(np.float32))

    #Reshape target tensors for BCELoss compatibility
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1, 1)

    #Create DataLoader for batch training
    batch_size = 512
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Initialize the model
    model = LogisticRegression(input_feat = X_train.shape[1])

    #Define loss function
    loss = nn.BCELoss()
    optimiser = torch.optim.SGD(params = model.parameters(), lr=0.01)

    for epoch in tqdm(range(n_iterations), desc="Training Progress", unit="epoch", dynamic_ncols=True):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:

            #forward pass
            y_preds = model(batch_X)
            L = loss(y_preds, batch_y)

            #Backprop
            L.backward()

            #Update parameters
            optimiser.step()

            #Zero gradients
            optimiser.zero_grad()
            
            epoch_loss += L.item()

        #Print loss at intervals 
        if epoch % 100 == 0 or epoch == n_iterations - 1:
            print(f'Epoch {epoch}, Loss {epoch_loss:.3f}')

    print("\n")

    #Evaluate model
    with torch.no_grad():
        #Predict probabilities
        y_prob = model(X_test)
        y_pred = (y_prob >= 0.5).float()

        #Convert to NumPy
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        y_prob_np = y_prob.cpu().numpy()    

        #Calculate metrics
        accuracy = accuracy_score(y_test_np, y_pred_np)
        balanced_acc = balanced_accuracy_score(y_test_np, y_pred_np)
        classif_rep = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test_np, y_prob_np)
        roc_pr = average_precision_score(y_test_np, y_prob_np)

        plot_roc_pr(y_test_np, y_prob_np)

        #Print results  
        # print("\nTraining Completed\n")
        # print(f'Accuracy: {accuracy:.2f} %')
        #print("Classification Report:")
        #print(classification_report(y_test_np, y_pred_np))
        # print(f"Balanced Accuracy: {balanced_acc:.3f}")
        # print(f"ROC-AUC Score: {roc_auc:.3f}")

    return accuracy, balanced_acc, roc_auc, classif_rep, roc_pr

#Tune Hyperparameters (Using GridSearchCV)
def tune_logistic_regression():
    #Define hyperparameter grid
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'batch_size': [128, 512, 1024],
        'n_iterations': [100, 300, 500],
        'optimizer_type': ['SGD', 'Adam'],
        'weight_decay': [0.0, 0.0001, 0.001]
    }

    best_model = None
    best_params = None
    best_accuracy = 0
    best_balanced_acc = 0

    #Iterate through all combinations 
    for params in ParameterGrid(param_grid):
        # Train model with given hyperparameters
        accuracy, balanced_acc, roc_auc, _ = train_logistic_regression(
            n_iterations=params["n_iterations"],
            batch_size=params["batch_size"]
        )

    # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_balanced_acc = balanced_acc
            best_params = params
            best_model = train_logistic_regression(n_iterations=params["n_iterations"], batch_size=params["batch_size"])

    # Print Best Parameters
    # print("\n Best Hyperparameters:")
    # print(best_params)
    # print(best_accuracy)
    # print(f"Best Accuracy: {best_accuracy:.4f}")
    # print(f"Best Balanced Accuracy: {best_balanced_acc:.4f}")

    return best_model, best_accuracy, best_params, best_balanced_acc