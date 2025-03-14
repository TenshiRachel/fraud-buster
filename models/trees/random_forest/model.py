from sklearn.ensemble import RandomForestClassifier


def get_rf_classifier(n_estimators, class_weight, max_depth, min_samples_leaf, max_features, random_state):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        warm_start=True,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state
    )
