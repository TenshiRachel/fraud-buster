from models.gradient_boost.adaboost.train import train_ada
from models.random_tree.decision_tree.train import train_decision_tree
from models.random_tree.decision_tree.train import tune_decision_tree
from src.eval import print_metrics
from models.gradient_boost.lightGBM.train import train_lgbm
from models.logistic_regression.train import train_logistic_regression
from src.eval import print_metrics, eval


def main():
    accuracies = []
    balances = []
    accuracies_feat = []
    balances_feat = []

    print("AdaBoost Model")
    ## AdaBoost Model
    # Feature Engineering = False
    accuracy, balanced = train_ada(feature_engineering=False)
    accuracies.append(accuracy)
    balances.append(balanced)
    print("Adaboost without feature engineering done")

    # Feature Engineering = True
    accuracy, balanced = train_ada(feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("Adaboost with feature engineering done")


    print("LightGBM Model Data")
    ## LightGBM Model
    # Feature Engineering = False
    accuracy, balanced = train_lgbm(feature_engineering=False)
    accuracies.append(accuracy)
    balances.append(balanced)
    print("LightGBM without feature engineering done")

    # Feature Engineering = True
    accuracy, balanced = train_lgbm(feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("LightGBM with feature engineering done")

    print("Decision Tree Model")
    train_decision_tree()
    # tune_decision_tree()

    print("Logistics Regression Model")
    logistic_acc = train_logistic_regression(feature_engineering=True, n_iterations=400)
    print(f'Logistic Regression Accuracy: {100 * logistic_acc:.2f} %')

    print_metrics(accuracies, balances, accuracies_feat, balances_feat)


if __name__ == '__main__':
    main()
