from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def adaboost_model(depth, num_iterations, learn_rate, random_state=42):
    base_estimator = DecisionTreeClassifier(max_depth=depth)
    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=num_iterations,
        learning_rate=learn_rate,
        random_state=random_state
    )

    return adaboost
