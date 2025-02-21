from src.process_data import get_train_data
from models.gradient_boost_adaboost.model import adaboost_model

# Hyperparameters
depth = 1
num_estimators = 50
learn_rate = 0.1


def train_ada():
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

    model = adaboost_model(depth=depth, num_estimators=num_estimators, learn_rate=learn_rate)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_prob