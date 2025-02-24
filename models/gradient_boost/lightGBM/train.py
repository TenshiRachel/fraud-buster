# from src.process_data import get_train_data
# from models.gradient_boost.lightGBM.model import lightgbm_model
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Hyperparameters
# max_depth = 6
# num_estimators = 100
# learning_rate = 0.1


# def train_lgbm():
#     X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

#     model = lightgbm_model(max_depth=max_depth, n_estimators=num_estimators, learning_rate=learning_rate)
    
#     # Train model
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]  # Probability for positive class

#     # Evaluation Metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_prob)

#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC-ROC: {roc_auc:.4f}")

#     return y_test, y_pred, y_prob

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
#     base_model = lightgbm_model()

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

#     print(f"\nBest Hyperparameters: {bayes_search.best_params_}")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"AUC-ROC: {roc_auc:.4f}")

#     return y_test, y_pred, y_prob, best_model

import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.process_data import get_train_data

# Load dataset
X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

def train_lgbm(trial=None, best_params=None):
    # If trial is None, use best_params if provided; otherwise, use default params
    if trial is not None:
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt'
        }
    else:
        params = best_params if best_params else {
            'max_depth': 6,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt'
        }

    # Train LightGBM model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc",
              callbacks=[lgb.early_stopping(30, verbose=False)])

    # Predict probabilities for AUC-ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    
    if trial is not None:
        return roc_auc_score(y_test, y_prob)
    
    # Predictions for evaluation
    y_pred = model.predict(X_test)
    return y_test, y_pred, y_prob

# Run optimization
study = optuna.create_study(direction='maximize')  # Maximize AUC-ROC
study.optimize(train_lgbm, n_trials=1)

# Check if trials were completed
if len(study.trials) > 0 and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
else:
    best_params = None
    print("Warning: No successful Optuna trials found. Using default parameters.")