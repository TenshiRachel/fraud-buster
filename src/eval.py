import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def plot_roc(fpr, tpr):
    # Measure actual positive against negatives misidentified as positives
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()


def eval(y_test, y_pred, y_prob, FIXED_FPR=0.05):
    fprs, tprs, thresholds = roc_curve(y_test, y_prob)
    # plot_roc(fprs, tprs)
    tpr = tprs[fprs < FIXED_FPR][-1]  # True positive rate
    fpr = fprs[fprs < FIXED_FPR][-1]  # False positive rate
    threshold = thresholds[fprs < FIXED_FPR][-1]

    # Summarizes area under curve, (1 good, 0.5 average, <0.5 bad)
    print("AUC:", roc_auc_score(y_test, y_prob))
    to_pct = lambda x: str(round(x, 4) * 100) + "%"

    print("=" * 50)
    print("TPR: ", to_pct(tpr), "\nFPR: ", to_pct(fpr), "\nThreshold: ", round(threshold, 2))

    # predictive_equality, disparities_df = get_fairness_metrics(y_test, y_prob, groups, FIXED_FPR)
    # print("Predictive Equality: ", to_pct(predictive_equality))

    print("=" * 50)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print("=" * 50)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f'Balanced accuracy: {balanced_accuracy * 100:.2f}%')

    print("=" * 50)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Precision: {precision}")

    print("=" * 50)
    print('Classification report')
    print(classification_report(y_test, y_pred, zero_division=0))


def print_metrics(baselines, balanced_baselines, balanced, balanced_balanced):
    header_list = [" ", "ADABOOST", "LIGHTGBM", "RANDOM FOREST", "LOGISTIC REGRESSION"]

    baselines.insert(0, 'Baseline accuracy')
    balanced_baselines.insert(0, 'Balanced baseline accuracy')
    balanced.insert(0, 'Balanced accuracy')
    balanced_balanced.insert(0, 'Balanced balanced accuracy')

    full = [header_list, balanced_balanced, balanced_baselines, balanced, baselines]
    ####### Add in precision, roc_auc comparison between models

    df = pd.DataFrame(full)

    print("=" * 50)
    print('Summary of model accuracies:')
    print(df)
