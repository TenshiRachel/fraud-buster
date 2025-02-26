from sklearn.tree import DecisionTreeClassifier

def decision_tree_model(max_depth, max_features, min_samples_leaf, min_samples_split, random_state):
    # Initialize and train the Decision Tree model
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,          # Limit depth to prevent overfitting
        max_features=max_features,   # Use square root of features to limit complexity
        min_samples_leaf=min_samples_leaf,    # Set minimum leaf size to avoid small leaf nodes
        min_samples_split=min_samples_split,   # Require more samples to make a split
        random_state=random_state,
        class_weight='balanced'
    )
    return model
