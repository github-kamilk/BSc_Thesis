import sys
from datetime import datetime

from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.SemiBoost import SemiBoost
from LAMDA_SSL.Algorithm.Classification.LapSVM import LapSVM
from LAMDA_SSL.Algorithm.Classification.VAT import VAT
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Algorithm.Classification.LabelPropagation import LabelPropagation
from LAMDA_SSL.Algorithm.Classification.TemporalEnsembling import TemporalEnsembling
from LAMDA_SSL.Algorithm.Classification.MeanTeacher import MeanTeacher


from data_preprocessing import preprocess_cifar10, preprocess_imdb, preprocess_breast_cancer
from model_training import train_and_evaluate
from utils import split_labeled_unlabeled, save_results, knn_factory, random_forest_factory, linear_svc_factory, gaussianNB_factory, decision_tree_factory, svc_factory


def train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size,
                models_hyperparameters, folder_to_save_results):
    print(f"Alghoritm: {model_name}")
    print(f"Dataset: {dataset_name}")
    X_train, y_train, X_test, y_test = preprocess_function()
    X_labeled, y_labeled, X_unlabeled, y_unlabeled = split_labeled_unlabeled(X_train, y_train, labeled_size)
    print(f"X_labeled shape: {X_labeled.shape}, X_unlabeled shape: {X_unlabeled.shape}")
    print(f"y_labeled shape: {y_labeled.shape}")
    
    dataset_specific_params = models_hyperparameters.get(model_name, {}).get(dataset_name, {})
    model_params = {k: v() if callable(v) else v for k, v in dataset_specific_params.items()}
    model = model_function(**model_params)

    # results[model_name][dataset_name] = train_and_evaluate(dataset_name, model, X_labeled, y_labeled, X_unlabeled, X_test, y_test)    
    train_results, transductive_results, test_results = train_and_evaluate(model_name, model_params, dataset_name, model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test)
    save_results(train_results, transductive_results, test_results, model_name, dataset_name, model_params, labeled_size, folder_to_save_results)
        
    print(f"{dataset_name} Results:", test_results)
    print()

if __name__ == "__main__":
    datasets = {
                'cifar10' : preprocess_cifar10, 
                'imdb' : preprocess_imdb, 
                'breast_cancer': preprocess_breast_cancer
                }

    models = {
           'Tri_Training' : Tri_Training,
            'Assemble' : Assemble,
            'SemiBoost' : SemiBoost,
            #'LapSVM' : LapSVM,
           # 'TSVM' : TSVM,
            'LabelPropagation' : LabelPropagation
            }

    models_hyperparameters = {
    'Tri_Training': {
        'cifar10': {
            'base_estimator': random_forest_factory,
            'base_estimator_2': knn_factory,
            'base_estimator_3': decision_tree_factory
        },
        'imdb': {
            'base_estimator': random_forest_factory,
            'base_estimator_2': linear_svc_factory,
            'base_estimator_3': gaussianNB_factory
        },
        'breast_cancer': {
            'base_estimator': random_forest_factory,
            'base_estimator_2': linear_svc_factory,
            'base_estimator_3': gaussianNB_factory
        }
    },
    'Assemble': {
        'cifar10': {
            'base_estimator' : random_forest_factory,
            'T' : 50    
        },
        'imdb': {
            'base_estimator' : random_forest_factory,
            'T' : 50
        },
        'breast_cancer': {
            'base_estimator' : random_forest_factory,
            'T' : 50
        }
    },
    'SemiBoost': {
        'cifar10': {
            'base_estimator' : gaussianNB_factory,
            'T' : 100    
        },
        'imdb': {
            'base_estimator' : gaussianNB_factory,
            'T' : 50
        },
        'breast_cancer': {
            'base_estimator' : linear_svc_factory,
            'T' : 300
        }
    },
    'LapSVM': {
        'cifar10': {
            'distance_function' : ['rbf', 'linear', 'knn'],
            'kernel_function' : ['rbf', 'linear'],
            'neighbor_mode' : 'distance'    
        },
        'imdb': {
            'distance_function' : ['rbf', 'linear', 'knn'],
            'kernel_function' : ['rbf', 'linear'],
            'neighbor_mode' : 'distance'
        },
        'breast_cancer': {
            'distance_function' : ['rbf', 'linear', 'knn'],
            'kernel_function' : ['rbf', 'linear'],
            'neighbor_mode' : 'distance'
        }
    },
    'TSVM': {
        'cifar10': {
            'max_iter' : [3*10**4, 10*10**4, 50*10**4, 100*10**4, 150*10**4, 300*10**4],
            'kernel' : ['rbf', 'linear'],
            'random_state' : 42    
        },
        'imdb': {
            'max_iter' : [3*10**4, 10*10**4, 50*10**4, 100*10**4, 150*10**4, 300*10**4],
            'kernel' : ['rbf', 'linear'],
            'random_state' : 42
        },
        'breast_cancer': {
            'max_iter' : [3*10**4, 10*10**4, 50*10**4, 100*10**4, 150*10**4, 300*10**4],
            'kernel' : ['rbf', 'linear'],
            'random_state' : 42
        }
    },
    'LabelPropagation': {
        'cifar10': {
            'kernel' : 'knn',
            'max_iter' : 100    
        },
        'imdb': {
            'kernel' : 'rbf',
            'max_iter' : 100
        },
        'breast_cancer': {
            'kernel' : 'knn',
            'max_iter' : 100
        }
    }
}

    folder_to_save_results = "experiments_results_base_parameters_small_labeled_size"
    
    for model_name, model_function in models.items():
        for dataset_name, preprocess_function in datasets.items():
            for labeled_size in [.03, .05, .07, .1, .12, .15, .20, .25, .30, .35, .4, .5, .6, .7, .8, .9]:
                train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size, models_hyperparameters, folder_to_save_results)

