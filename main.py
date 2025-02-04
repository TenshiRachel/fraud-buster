from models.gradient_boost.train import train_ada
from src.eval import eval


def main():
    ada_y_test, ada_y_pred, ada_y_prob = train_ada()
    eval(y_test=ada_y_test, y_pred=ada_y_pred, y_prob=ada_y_prob)


if __name__ == '__main__':
    main()
