from models.gradient_boost.adaboost.train import train_ada
from models.random_tree.decision_tree.train import train_decision_tree
from models.random_tree.random_forest.train import train_rf
from src.eval import print_metrics
from models.gradient_boost.lightGBM.train import train_lgbm
from models.logistic_regression.train import train_logistic_regression
from src.eval import print_metrics, eval


def main():
    accuracies = []
    balances = []
    accuracies_feat = []
    balances_feat = []

    print(" >>> AdaBoost Model")
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



    print(" >>> LightGBM Model Data")
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



    print(" >>> Random Forest Model")
    # Feature Engineering = False
    accuracy, balanced = train_rf(feature_engineering=False)
    accuracies.append(accuracy)
    balances.append(balanced)
    print("Random Forest without feature engineering done")

    # Feature Engineering = True
    accuracy, balanced = train_rf(feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("Random Forest with feature engineering done")



    print(" >>> Decision Tree Model")
    train_decision_tree()
    # tune_decision_tree()

    print("Logistic Regression Results:")
    n_iterations = 50
    accuracy, balanced_acc, roc_auc, classifi_rep = train_logistic_regression(feature_engineering=True, n_iterations=50, batch_size=512)
    print(f"Number of iterations: {n_iterations}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classifi_rep)


    print_metrics(accuracies, balances, accuracies_feat, balances_feat)


if __name__ == '__main__':
    main()
