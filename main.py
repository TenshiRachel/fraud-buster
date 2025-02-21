from models.gradient_boost_adaboost.train import train_ada
from models.gradient_boost_LightGBM.train import train_lgbm
from src.eval import eval


def main():
    print("AdaBoost Model Data")
    ada_y_test, ada_y_pred, ada_y_prob = train_ada()
    eval(y_test=ada_y_test, y_pred=ada_y_pred, y_prob=ada_y_prob)

    # print("LightGBM Model Data")
    # lgbm_y_test, lgbm_y_pred, lgbm_y_prob = train_lgbm()
    # eval(y_test=lgbm_y_test, y_pred=lgbm_y_pred, y_prob=lgbm_y_prob)


if __name__ == '__main__':
    main()
