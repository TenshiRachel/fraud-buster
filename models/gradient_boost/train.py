from src.process_data import get_train_data
from models.gradient_boost.model import adaboost_model
from models.gradient_boost.eval import eval_ada

# Hyperparameters
depth = 1
num_iterations = 50
learn_rate = 0.1


def train():
    X_train, X_test, y_train, y_test = get_train_data(test_size=0.2, random_state=42)

    model = adaboost_model(depth=depth, num_iterations=num_iterations, learn_rate=learn_rate)

    model.fit(X_train, y_train)
    eval_ada(model, X_test, y_test)


if __name__ == '__main__':
    train()
