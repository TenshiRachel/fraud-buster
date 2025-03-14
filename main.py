from models.gradient_boost.adaboost.train import train_ada
from models.random_tree.decision_tree.train import train_decision_tree
from models.random_tree.decision_tree.train import tune_decision_tree
from models.random_tree.random_forest.train import train_rf
from src.eval import print_metrics
from models.gradient_boost.lightGBM.train import train_lgbm , find_best_hyperparameters
from models.logistic_regression.train import train_logistic_regression
from models.neural_network.train import train_nn
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
    accuracy, balanced = train_rf(n_estimators=200, max_depth=15, min_samples_leaf=15, max_features="sqrt", class_weight="balanced", feature_engineering=False)
    accuracies.append(accuracy)
    balances.append(balanced)
    print("Random Forest without feature engineering done")

    # Feature Engineering = True
    accuracy, balanced = train_rf(n_estimators=200, max_depth=15, min_samples_leaf=15, max_features="sqrt", class_weight="balanced", feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("Random Forest with feature engineering done")


    print(" >>> Decision Tree Model")
    accuracy, balanced = train_decision_tree(feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("Decision Tree with feature engineering done")
    
    accuracy, balanced = train_decision_tree(feature_engineering=False)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("Decision Tree without feature engineering done")
    print("Logistic Regression Results:")
    n_iterations = 100
    accuracy, balanced_acc, roc_auc, classifi_rep, roc_pr = train_logistic_regression(feature_engineering=False, n_iterations=n_iterations, batch_size=512)
    print(f"Number of iterations: {n_iterations}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    print(f"ROC-PR Score: {roc_pr:.4f}")
    print("Classification Report:")
    print(classifi_rep)


    print(" >>> Neural Network Model")
    ## AdaBoost Model
    # Feature Engineering = False
    accuracy, balanced = train_nn(feature_engineering=False)
    accuracies.append(accuracy)
    balances.append(balanced)
    print("Neural Network without feature engineering done")

    # Feature Engineering = True
    accuracy, balanced = train_nn(feature_engineering=True)
    accuracies_feat.append(accuracy)
    balances_feat.append(balanced)
    print("Neural Network with feature engineering done")

    print_metrics(accuracies, balances, accuracies_feat, balances_feat)


if __name__ == '__main__':
    main()
