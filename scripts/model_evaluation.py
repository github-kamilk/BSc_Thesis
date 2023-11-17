from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    print(f"Testing model")
    y_pred = model.predict(X_test)
    if hasattr(model.predict, '__code__') and 'Transductive' in model.predict.__code__.co_varnames:
        y_pred = model.predict(X=X_test, Transductive=False)
    else:
        y_pred = model.predict(X_test)
   # print(y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }