from src.process_data import get_train_data
from models.gradient_boost_adaboost.model import adaboost_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_score

# Hyperparameters
depth = 3
num_estimators = 100
learn_rate = 0.1


def train_ada(feature_engineering=False):
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.25, random_state=42,
                                                      feature_engineering=feature_engineering)

    model = adaboost_model(depth=depth, num_estimators=num_estimators, learn_rate=learn_rate)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # print("=" * 50)
    # precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    # print(f"Precision: {precision}")
    #
    # print("=" * 50)
    # print('Classification report')
    # print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy, balanced_accuracy
