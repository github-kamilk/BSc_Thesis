import time
import scipy
from model_evaluation import evaluate_model

def train_and_evaluate(dataset_name, model, X_train, y_train, X_unlabeled, X_test, y_test):
    print(f"Training model")
        
    if dataset_name == 'cifar10':
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_unlabeled = X_unlabeled.reshape(X_unlabeled.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Convert sparse arrays to dense arrays for IMDB data
    elif dataset_name == 'imdb':
        X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
        X_unlabeled = X_unlabeled.toarray() if scipy.sparse.issparse(X_unlabeled) else X_unlabeled
        X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test

    print(f"X_train shape: {X_train.shape}, X_unlabeled shape: {X_unlabeled.shape}")
    print(f"y_train shape: {y_train.shape}")
    start_time = time.time()
    model.fit(X=X_train, y=y_train, unlabeled_X=X_unlabeled)
    training_time = time.time() - start_time
    results = evaluate_model(model, X_test, y_test)
    results['time'] = training_time
    return results