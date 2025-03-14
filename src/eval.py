def print_metrics(accuracy, balanced_accuracy, roc, roc_pr, class_report):
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")

    print("=" * 50)
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    print("=" * 50)
    print(f"AUC-ROC: {roc:.4f}")

    print("=" * 50)
    print(f"ROC-PR-AUC: {roc_pr:.4f}")

    print("=" * 50)
    print('Classification report')
    print(class_report)
