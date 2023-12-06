import time
import joblib
import scipy
import datetime
from model_evaluation import evaluate_model

def format_hyperparams_for_filename(hyperparams):
    safe_hyperparams = []
    for key, value in hyperparams.items():
        if callable(value):
            value = value.__name__
        safe_value = str(value).replace('<', '').replace('>', '').replace(' ', '_').replace(':', '').replace('\\', '').replace('/', '')
        safe_hyperparams.append(f"{key}_{safe_value}")
    return "_".join(safe_hyperparams)

def train_and_evaluate(model_name, hyperparams, dataset_name, model, X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test):
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


     # Calculating statistics for the training set
    train_results = evaluate_model(model, X_train, y_train, 'train')

     # Calculating statistics for the transductive set
    transductive_results = evaluate_model(model, X_unlabeled, y_unlabeled, 'transductive')

    # Calculating statistics for the test set
    test_results = evaluate_model(model, X_test, y_test, 'test')
    test_results['training_time'] = training_time

    # Saving the model
    # current_time = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    # hyperparams_str = format_hyperparams_for_filename(hyperparams)
    # model_filename = f"models/{model_name}_{dataset_name}_{hyperparams_str}_{current_time}_model.joblib"
    # try:
    #     joblib.dump(model, model_filename)
    #     print(f"Model saved as {model_filename}")
    # except Exception as e:
    #     print(f"Error during saving model {model_name} with params {hyperparams}: {e}")


    return train_results, transductive_results, test_results