from sklearn.ensemble import RandomForestClassifier

def get_rf_classifier(n_estimators, class_weight, max_depth, min_samples_leaf, max_features, random_state):
    return RandomForestClassifier(
        n_jobs=-1,  # Use all CPU cores
        warm_start=True,  # Incremental training
        n_estimators=500,  # more trees for better stability
        random_state=42,
        max_depth=15,  # more depth = longer training, lower value = lower overfitting
        min_samples_leaf=15,  # lower value = lower variance, higher value = more minimum leaves = less overfitting
        max_features="log2",  # use sqrt features per split | log2 or sqrt
        class_weight="balanced"  # adjust class weight impact | balanced or balanced_subsample
    )