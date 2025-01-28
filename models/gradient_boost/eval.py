from sklearn.metrics import accuracy_score, classification_report, precision_score


def eval_ada(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Precision: {precision}")

    print('Classification report')
    print(classification_report(y_test, y_pred, zero_division=0))
