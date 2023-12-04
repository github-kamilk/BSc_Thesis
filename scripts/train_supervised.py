import sys
from datetime import datetime
import time
import joblib
import scipy
import datetime
from model_evaluation import evaluate_model

from data_preprocessing import preprocess_cifar10, preprocess_imdb, preprocess_breast_cancer
from utils import split_labeled_unlabeled, save_results
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def train_and_evaluate(dataset_name, model, X_train, y_train, X_test, y_test):
    print(f"Training model")
        
    if dataset_name == 'cifar10':
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Convert sparse arrays to dense arrays for IMDB data
    elif dataset_name == 'imdb':
        X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
        X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    start_time = time.time()
    model.fit(X=X_train, y=y_train)
    training_time = time.time() - start_time


     # Calculating statistics for the training set
    train_results = evaluate_model(model, X_train, y_train, 'train')

    # Calculating statistics for the test set
    test_results = evaluate_model(model, X_test, y_test, 'test')
    test_results['training_time'] = training_time

    return train_results, test_results

def train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size,
                models_hyperparameters, folder_to_save_results):
    print(f"Alghoritm: {model_name}")
    print(f"Dataset: {dataset_name}")
    X_train, y_train, X_test, y_test = preprocess_function()
    if labeled_size != 1.0:
        X_labeled, y_labeled, _ = split_labeled_unlabeled(X_train, y_train, labeled_size, random_state=None)
    else:
        X_labeled = X_train
        y_labeled = y_train
    print(f"X_labeled shape: {X_labeled.shape}")
    print(f"y_labeled shape: {y_labeled.shape}")

    try:
        model_params = models_hyperparameters.get(model_name, {})
        model_params = {k: v() if callable(v) else v for k, v in model_params.items()}
        model = model_function(**model_params)
    except Exception as e:
        print(f"Error during model initialization for {model_name} with params {model_params}: {e}")
            
    try:
        train_results, test_results = train_and_evaluate(dataset_name, model, X_labeled, y_labeled, X_test, y_test)
    except Exception as e:
        print(f"Error during training/evaluation for {model_name} with params {model_params}: {e}")
        
    try:
        save_results(train_results, test_results, model_name, dataset_name, model_params, labeled_size, folder_to_save_results)
    except Exception as e:
        print(f"Error during saving results for {model_name} with params {model_params}: {e}")
        
    print(f"{dataset_name} Results:", test_results)
    print()


if __name__ == "__main__":
    datasets = {
                'cifar10' : preprocess_cifar10, 
                'imdb' : preprocess_imdb, 
                'breast_cancer': preprocess_breast_cancer
                }

    models = {
            'RandomForestClassifier': RandomForestClassifier
           # 'DecisionTreeClassifier': DecisionTreeClassifier,
          #  'LinearSVC': LinearSVC,
         #   'GaussianNB': GaussianNB,
        #    'KNeighborsClassifier': KNeighborsClassifier
            }

    models_hyperparameters = {
                'RandomForestClassifier' : {
                }
                ,'DecisionTreeClassifier' : {
                }
                ,'LinearSVC' : {
                    'max_iter': 10**4
                }
                ,'GaussianNB' : {
                }
                ,'KNeighborsClassifier' : {
                    'n_neighbors' : 5
                }
                , "SVC" : {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'probability': True,
                    'gamma': 'auto'
                }
                }
    
    results = {
        'RandomForestClassifier' : {},
        'DecisionTreeClassifier' : {},
        'LinearSVC' : {},
        'GaussianNB' : {},
        'KNeighborsClassifier' : {}
    }

   # labeled_size = 0.3
    folder_to_save_results = "experiments_results_supervised"
    for model_name, model_function in models.items():
        for dataset_name, preprocess_function in datasets.items():
            for labeled_size in [0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9, 1.0]:
                num_of_iteration = 50 if labeled_size < 0.3 else 10
                for _ in range(num_of_iteration):
                    #REMEMBER TO CHANGE RANDOM SEED!!!!!!
                    train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size, models_hyperparameters, folder_to_save_results)


