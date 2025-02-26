from src.process_data import get_train_data
from models.gradient_boost.lightGBM.model import lightgbm_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb


# Hyperparameters
max_depth = 10 
num_leaves = 180 
num_estimators = 500 
learning_rate = 0.1

def train_lgbm(feature_engineering):
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=feature_engineering) 

    model = lightgbm_model(max_depth=max_depth, num_leaves = num_leaves, n_estimators=num_estimators, learning_rate=learning_rate) 
    
    # Train model
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class

    # Accuracy Metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    ## Evaluation Metrics
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # roc_auc = roc_auc_score(y_test, y_prob)

    ## Print metrics
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"AUC-ROC: {roc_auc:.4f}")

    return accuracy, balanced_accuracy


# Hyperparameter tuning
def find_best_hyperparameters():
    # Load dataset with feature engineering enabled
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.25, random_state=42, feature_engineering=True)

    # Define LightGBM classifier
    lgbm = lgb.LGBMClassifier(objective='binary', class_weight='balanced', random_state=42, n_jobs=-1)

    # Define a smaller hyperparameter grid (including scale_pos_weight)
    # param_grid = {
    #     'max_depth': [4, 6, 8, 10, 12],
    #     'learning_rate': [0.01, 0.1],  
    #     'n_estimators': [100, 200, 500],
    #     'scale_pos_weight': np.linspace(10, 30, 5).tolist()  # Generates 5 values between 10 and 50
    # }

    param_grid = {
        'num_leaves': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9, 12],
        'learning_rate': [0.01, 0.1],  
        'n_estimators': [100, 200, 500],
        'scale_pos_weight': [10, 30 , 50]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2,
                               scoring='balanced_accuracy')

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Print best hyperparameters
    print("Best Hyperparameters: ", grid_search.best_params_)
    
    return grid_search.best_params_




# Best Hyperparameters:  {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 500, 'scale_pos_weight': 10.0}
# Best Hyperparameters:  {'learning_rate': 0.1, 'n_estimators': 200, 'num_leaves': 64, 'scale_pos_weight': 10.0}
# Best Hyperparameters:  {'learning_rate': 0.1, 'max_depth': 9, 'num_leaves': 100, 'n_estimators': 500, 'scale_pos_weight': 10.0}
# Best Hyperparameters: OrderedDict({'feature_fraction': 0.6681937186444209, 'lambda_l1': 0.251546610771656, 'lambda_l2': 0.06094157054671746, 'learning_rate': 0.04229814073762352, 'max_bin': 116, 'max_depth': 3, 'min_data_in_leaf': 24, 'n_estimators': 435, 'num_leaves': 36, 'scale_pos_weight': 18.388548900590937



# from src.process_data import get_train_data
# from models.gradient_boost.lightGBM.model import lightgbm_model
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from skopt import BayesSearchCV
# import lightgbm as lgb
# from skopt.space import Real, Integer

# # Define hyperparameter search space
# param_space = {
#     'num_leaves': Integer(16, 64),  # Integer range
#     'max_depth': Integer(3, 12),  # Increased max depth
#     'learning_rate': Real(0.01, 0.2, prior='log-uniform'),  # Log-uniform for continuous values
#     'n_estimators': Integer(100, 500),  # Integer range
#     'lambda_l1': Real(0.0, 1.0),  # Float range
#     'lambda_l2': Real(0.0, 1.0),  # Float range
#     'min_data_in_leaf': Integer(5, 50),  # Integer range
#     'scale_pos_weight': Real(10, 50),  # Float range for imbalanced datasets
#     'feature_fraction': Real(0.5, 1.0),  # Float range for feature selection
#     'max_bin': Integer(64, 255)  # Integer range for feature discretization
# }

# def train_lgbm():
#     # Load training data
#     X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

#     # Debug: Print data shapes
#     print(f"Training data shape: {X_train.shape}, {y_train.shape}")
#     print(f"Test data shape: {X_test.shape}, {y_test.shape}")

#     # Initialize LightGBM model
#     base_model = lightgbm_model(num_leaves=num_leaves, n_estimators=num_estimators, learning_rate=learning_rate) 

#     # Perform Bayesian Optimization
#     bayes_search = BayesSearchCV(
#         estimator=base_model,  # Use the initialized model
#         search_spaces=param_space,
#         n_iter=5,  # Try at least 30 different hyperparameter sets
#         cv=3,
#         scoring='roc_auc',
#         n_jobs=-1,  # Use all available cores
#         verbose=1,
#         error_score=0  # Prevent errors from stopping the search
#     )

#     # Debug: Print search space
#     print(f"Search space: {param_space}")

#     # Train with hyperparameter tuning
#     bayes_search.fit(X_train, y_train)

#     # Get best model with tuned parameters
#     best_model = bayes_search.best_estimator_

#     # Predictions
#     y_pred = best_model.predict(X_test)
#     y_prob = best_model.predict_proba(X_test)[:, 1]  # Probability for positive class

#     # Evaluation Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob)

#     print(f"\nBest Hyperparameters: {bayes_search.best_params_}") #Best Hyperparameters: OrderedDict({'feature_fraction': 0.6681937186444209, 'lambda_l1': 0.251546610771656, 'lambda_l2': 0.06094157054671746, 'learning_rate': 0.04229814073762352, 'max_bin': 116, 'max_depth': 3, 'min_data_in_leaf': 24, 'n_estimators': 435, 'num_leaves': 36, 'scale_pos_weight': 18.388548900590937
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC-ROC: {roc_auc:.4f}")

#     return y_test, y_pred, y_prob, best_model