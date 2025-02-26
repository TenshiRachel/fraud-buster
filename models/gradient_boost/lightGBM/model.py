import lightgbm as lgb

def lightgbm_model(max_depth, num_leaves, n_estimators, learning_rate, random_state=42, scale_pos_weight=10):
    model = lgb.LGBMClassifier(
        boosting_type = 'gbdt',  # Gradient boosting decision tree
        objective = 'binary',  # For binary classification
        max_depth = max_depth,
        num_leaves = num_leaves, 
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        scale_pos_weight = scale_pos_weight,  # Handles class imbalance
        random_state = random_state,
        n_jobs = -1  # Use all CPU cores
        # verbosity = 1  # Suppress warnings (set to 1 for debugging)
    )
    return model