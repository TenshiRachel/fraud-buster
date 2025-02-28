from sklearn.ensemble import RandomForestClassifier

def get_rf_classifier(n_estimators, class_weight, max_depth, min_samples_leaf, max_features, random_state):
    return RandomForestClassifier(
        n_estimators=n_estimators,  # more trees for better stability
        class_weight=class_weight,  # adjust class weight impact (either balanced or balanced_subsample)
        max_depth=max_depth,  # more depth = longer training, lower value = lower overfitting
        min_samples_leaf=min_samples_leaf,  # lower value = lower variance, higher value = more minimum leaves = less overfitting
        max_features=max_features,  # use sqrt features per split
        random_state=random_state,
        warm_start=True,  # Incremental training
        n_jobs=-1,  # Use all CPU cores
    )