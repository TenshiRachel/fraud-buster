from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src.process_data import get_train_data
from src.eval import print_metrics
from models.gradient_boost.adaboost.model import adaboost_model
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, classification_report,
                             roc_auc_score, recall_score, f1_score, average_precision_score)
from sklearn.model_selection import GridSearchCV

# Hyperparameters
depth = 3
num_estimators = 50
learn_rate = 0.01

param_grid = {
    'estimator__max_depth': [2, 3],  # Example values for depth
    'n_estimators': [50, 100],  # Example values for number of estimators
    'learning_rate': [0.01, 0.1]  # Example values for learning rate
}


def train_ada(feature_engineering=False):
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42,
                                                      feature_engineering=feature_engineering)

    model = adaboost_model(depth=depth, num_estimators=num_estimators, learn_rate=learn_rate)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    roc_pr = average_precision_score(y_test, y_prob)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print_metrics(accuracy, balanced_accuracy, roc, roc_pr, class_report)


def find_best_hyperparameters():
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.25, random_state=42,
                                                      feature_engineering=True)

    base_estimator = DecisionTreeClassifier(class_weight='balanced', max_depth=depth)
    adaboost = AdaBoostClassifier(estimator=base_estimator, random_state=42, n_estimators=num_estimators, learning_rate=learn_rate)

    grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2,
                               scoring='balanced_accuracy')

    # Fit the model with GridSearch
    grid_search.fit(X_train, y_train)

    # Best hyperparameters found by GridSearch
    print("Best Hyperparameters: ", grid_search.best_params_)
