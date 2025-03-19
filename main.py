from models.gradient_boost.adaboost.train import train_ada
from models.trees.decision_tree.train import train_decision_tree
from models.trees.random_forest.train import train_rf
from models.gradient_boost.lightGBM.train import train_lgbm
from models.logistic_regression.train import train_logistic_regression
from models.neural_network.train import train_nn


def main():
    # print(" >>> AdaBoost Model")
    # # AdaBoost Model
    # # Feature Engineering = False
    # train_ada(feature_engineering=False)
    # print("Adaboost without feature engineering done")
    #
    # # Feature Engineering = True
    # train_ada(feature_engineering=True)
    # print("Adaboost with feature engineering done")
    #
    # print(" >>> LightGBM Model Data")
    # # Feature Engineering = False
    # train_lgbm(feature_engineering=False)
    # print("LightGBM without feature engineering done")
    #
    # # Feature Engineering = True
    # train_lgbm(feature_engineering=True)
    # print("LightGBM with feature engineering done")
    #
    # print(" >>> Random Forest Model")
    # # Feature Engineering = False
    # train_rf(feature_engineering=False)
    # print("Random Forest without feature engineering done")
    #
    # # Feature Engineering = True
    # train_rf(feature_engineering=True)
    # print("Random Forest with feature engineering done")
    #
    # print(" >>> Decision Tree Model")
    # train_decision_tree(feature_engineering=True)
    # print("Decision Tree with feature engineering done")
    #
    # train_decision_tree(feature_engineering=False)
    # print("Decision Tree without feature engineering done")

    print(" >>> Logistic Regression")
    # train_logistic_regression(feature_engineering=False, batch_size=512)
    # print("Logistic Regression without feature engineering done")

    train_logistic_regression(feature_engineering=True, batch_size=512)
    print("Logistic Regression with feature engineering done")

    # print(" >>> Neural Network Model")
    # # Feature Engineering = False
    # train_nn(feature_engineering=False)
    # print("Neural Network without feature engineering done")
    #
    # # Feature Engineering = True
    # train_nn(feature_engineering=True)
    # print("Neural Network with feature engineering done")


if __name__ == '__main__':
    main()
