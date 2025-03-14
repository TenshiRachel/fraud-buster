from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from models.trees.decision_tree.model import decision_tree_model
from src.process_data import get_train_data
from src.eval import print_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# loading progress bar
from tqdm import tqdm


# Hyperparameters
max_depth = 10
max_features = 'sqrt'
min_samples_leaf = 1


def train_decision_tree(feature_engineering):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_engineering)
    
    print("Starting Decision Tree Model training process...")

    # Initialize and train the Decision Tree model
    model = decision_tree_model(max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf,
                                random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    # pr_auc = average_precision_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Print the results
    print_metrics(accuracy, balanced_accuracy, auc_roc, pr_auc, classification_rep)


def tune_decision_tree():
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=False)
    
    print("Finding best hyperparameters for Decision Tree Model...")

    # Initialize and train the Decision Tree model
    model = DecisionTreeClassifier(class_weight='balanced')
    
    # Define hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Output the best parameters and score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_}")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_score}")

    return best_model, test_score
