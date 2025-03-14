from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def adaboost_model(depth, num_estimators, learn_rate, random_state=42):
    # base_estimator = DecisionTreeClassifier(
    #     max_depth=depth,
    #     class_weight='balanced',
    #     random_state=random_state
    # )

    base_estimator = RandomForestClassifier(
        max_depth=depth,
        n_estimators=10,
        class_weight='balanced',
        random_state=random_state
    )

    # base_estimator = ExtraTreesClassifier(
    #     n_estimators=10,
    #     max_depth=depth,
    #     class_weight='balanced',
    #     random_state=random_state
    # )

    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=num_estimators,
        learning_rate=learn_rate,
        random_state=random_state
    )

    return adaboost
