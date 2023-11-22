from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X, y, dataset_type):
    print(f"Evaluating model on {dataset_type} data")
    y_pred = model.predict(X)
    if hasattr(model.predict, '__code__') and 'Transductive' in model.predict.__code__.co_varnames:
        y_pred = model.predict(X=X, Transductive=False)
    else:
        y_pred = model.predict(X)
    return {
        f"{dataset_type}_accuracy": accuracy_score(y, y_pred),
        f"{dataset_type}_precision": precision_score(y, y_pred),
        f"{dataset_type}_recall": recall_score(y, y_pred),
        f"{dataset_type}_f1_score": f1_score(y, y_pred)
    }