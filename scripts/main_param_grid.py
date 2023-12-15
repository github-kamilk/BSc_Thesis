import itertools

from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.SemiBoost import SemiBoost
from LAMDA_SSL.Algorithm.Classification.LapSVM import LapSVM
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Algorithm.Classification.LabelPropagation import LabelPropagation


from data_preprocessing import preprocess_cifar10, preprocess_imdb, preprocess_breast_cancer
from model_training import train_and_evaluate
from utils import split_labeled_unlabeled, save_results, knn_factory, random_forest_factory, linear_svc_factory, gaussianNB_factory, decision_tree_factory, svc_factory


def generate_hyperparameter_combinations(hyperparams):
    keys = hyperparams.keys()
    values = (hyperparams[key] if isinstance(hyperparams[key], list) else [hyperparams[key]] for key in keys)
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size,
                models_hyperparameters, folder_to_save_results):
    print(f"Alghoritm: {model_name}")
    print(f"Dataset: {dataset_name}")
    X_train, y_train, X_test, y_test = preprocess_function()
    X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_labeled_unlabeled(X_train, y_train, labeled_size)
    print(f"X_labeled shape: {X_labeled.shape}, X_unlabeled shape: {X_unlabeled.shape}")
    print(f"y_labeled shape: {y_labeled.shape}")

    for hyperparams in generate_hyperparameter_combinations(models_hyperparameters[model_name]):
        print(f"Hyperparameters: {hyperparams}")

        try:
            model_params = {k: v() if callable(v) else v for k, v in hyperparams.items()}
            model = model_function(**model_params)
        except Exception as e:
            print(f"Error during model initialization for {model_name} with params {hyperparams}: {e}")
            continue
        
        try:
            train_results, transductive_results, test_results = train_and_evaluate(model_name, hyperparams, dataset_name, model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test)
        except Exception as e:
            print(f"Error during training/evaluation for {model_name} with params {hyperparams}: {e}")
            continue

        try:
            save_results(train_results, transductive_results, test_results, model_name, dataset_name, model_params, labeled_size, folder_to_save_results)
        except Exception as e:
            print(f"Error during saving results for {model_name} with params {hyperparams}: {e}")
            continue

        print(f"{dataset_name} Results:", test_results)
        print()

if __name__ == "__main__":
    datasets = {
                'breast_cancer': preprocess_breast_cancer,
                'cifar10' : preprocess_cifar10, 
                'imdb' : preprocess_imdb
                }

    models = {
            # 'Tri_Training' : Tri_Training,
            # 'Assemble' : Assemble,
             'SemiBoost' : SemiBoost,
             'LapSVM' : LapSVM,
            #'TSVM' : TSVM
             'LabelPropagation' : LabelPropagation
            }

    models_hyperparameters = {
                'Tri_Training' : {
                    'base_estimator': random_forest_factory,
                    # 'base_estimator_2': linear_svc_factory,
                    'base_estimator_2': svc_factory,
                    'base_estimator_3': gaussianNB_factory
                                    }
                ,'Assemble' : {
                    # 'base_estimator' : [random_forest_factory, decision_tree_factory, linear_svc_factory, gaussianNB_factory],
                    'base_estimator' : svc_factory,
                    'T' : [3, 10, 50, 100, 150, 300]
                }
                ,'SemiBoost' : {
                    'base_estimator' : [random_forest_factory, decision_tree_factory, linear_svc_factory, gaussianNB_factory, knn_factory],
                    'similarity_kernel' : 'knn',
                    'T' : [3, 10, 50, 100, 150, 300]
                }
                ,'LapSVM' : {   
                    #'distance_function' : ['rbf', 'linear', 'knn'],
                    #'kernel_function' : ['rbf', 'linear'],
                    'gamma_k' : [10**-3, 10**3,]    
                }
                ,'TSVM' : {
                    'max_iter' : [3*10**4, 10*10**4, 50*10**4, 100*10**4, 150*10**4, 300*10**4],
                    'kernel' : ['rbf', 'linear'],
                    'random_state' : 42
                }
                ,'LabelPropagation' : {
                    'kernel' : 'rbf',# 'knn'],
                    'max_iter' : 50,
                    'gamma': [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**1, 10**2, 10**3, 10**4, 10**5]
                    #'n_neighbors' 
                }
                }
    
    labeled_size = 0.3
    folder_to_save_results = "experiments_results_all_parameters_tsvm"
    for model_name, model_function in models.items():
        for dataset_name, preprocess_function in datasets.items():
            train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size, models_hyperparameters, folder_to_save_results)