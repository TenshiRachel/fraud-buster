import lightgbm 

def lightgbm_model(max_depth, num_estimators, learning_rate, random_state=42, class_weight='balanced'):
    model = lightgbm.LGBMClassifier(
        boosting_type='gbdt',  # Gradient boosting decision tree
        objective='binary',  # For fraud detection (binary classification)
        max_depth=max_depth,
        n_estimators=num_estimators,
        learning_rate=learning_rate,
        class_weight=class_weight,  # Handles class imbalance
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbosity=-1  # Suppress warnings
    )

    return model