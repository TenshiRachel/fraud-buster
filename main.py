from models.gradient_boost.adaboost.train import train_ada
from models.gradient_boost.lightGBM.train import train_lgbm
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

    print_metrics(accuracies, balances, accuracies_feat, balances_feat)


if __name__ == '__main__':
    main()
