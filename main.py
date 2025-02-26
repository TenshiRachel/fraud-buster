from models.gradient_boost.adaboost.train import train_ada
from models.random_tree.decision_tree.train import train_decision_tree
from models.random_tree.decision_tree.train import tune_decision_tree
from models.gradient_boost.lightGBM import train
from src.eval import print_metrics


def main():
    accuracies = []
    balances = []
    accuracies_feat = []
    balances_feat = []

    print("AdaBoost Model")
    accuracy, balanced = train_ada(feature_engineering=False)
    accuracies.append(accuracy)
    balances.append(balanced)

    print("Adaboost without feature engineering done")

    accuracy, balanced = train_ada(feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)

    print("Adaboost with feature engineering done")

    print("LightGBM Model Data")
    lgbm_y_test, lgbm_y_pred, lgbm_y_prob = train_lgbm()
    eval(y_test=lgbm_y_test, y_pred=lgbm_y_pred, y_prob=lgbm_y_prob)
    
    print_metrics(accuracies, balances, accuracies_feat, balances_feat)

    print("Decision Tree Model")
    train_decision_tree()
    # tune_decision_tree()

if __name__ == '__main__':
    main()
