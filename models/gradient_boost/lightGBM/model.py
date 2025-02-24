import lightgbm as lgb

def lightgbm_model(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, scale_pos_weight=1):
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',  # Gradient boosting decision tree
        objective='binary',  # For binary classification
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,  # Handles class imbalance
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbosity=2  # Suppress warnings (set to 1 for debugging)
    )
    return model