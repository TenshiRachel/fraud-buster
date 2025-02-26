from src.process_data import get_train_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score
from imblearn.combine import SMOTETomek  # Improved balancing
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Use SMOTE + Tomek Links for better resampling
smote_tomek = SMOTETomek(random_state=42)

def train_rf():
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42, feature_engineering=True)
    print("got train data")

    # Apply SMOTE + Tomek
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

    rf_classifier = RandomForestClassifier(
        n_estimators=200,  # More trees for better stability
        class_weight="balanced_subsample",  # Adjust class weight impact
        max_depth=25,  # Prevent overfitting
        min_samples_leaf=5,  # Reduce variance
        max_features="sqrt",  # Use sqrt features per split
        random_state=42
    )

    print("Starting training...")
    for i in tqdm(range(1, n_estimators + 1), desc="Training Progress"):
        rf_classifier.n_estimators = i  # Incrementally increase trees
        rf_classifier.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # Print results
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)
    print(f"\nBalanced Accuracy: {balanced_accuracy:.4f}")
    print(f"\nAUC-ROC Score: {auc_roc:.4f}")  # Important for class imbalance

train_rf()
