from models.gradient_boost.adaboost.train import train_ada
<<<<<<< HEAD
from models.random_tree.decision_tree.train import train_decision_tree
from models.random_tree.decision_tree.train import tune_decision_tree
from models.gradient_boost.lightGBM import train
from src.eval import print_metrics
=======
from models.gradient_boost.lightGBM.train import train_lgbm
from src.eval import print_metrics, eval
>>>>>>> 9f296c753aa259b572ea5e369dd5c925824f5271


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

    print_metrics(accuracies, balances, accuracies_feat, balances_feat)

    print("Decision Tree Model")
    train_decision_tree()
    # tune_decision_tree()

if __name__ == '__main__':
    main()
