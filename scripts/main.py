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
from utils import split_labeled_unlabeled, save_results, knn_factory, random_forest_factory, linear_svc_factory, gaussianNB_factory


def train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size,
                models_hyperparameters, results):
    print(f"Alghoritm: {model_name}")
    print(f"Dataset: {dataset_name}")
    X_train, y_train, X_test, y_test = preprocess_function()
    X_labeled, y_labeled, X_unlabeled = split_labeled_unlabeled(X_train, y_train, labeled_size)
    print(f"X_labeled shape: {X_labeled.shape}, X_unlabeled shape: {X_unlabeled.shape}")
    print(f"y_labeled shape: {y_labeled.shape}")

    model_params = models_hyperparameters.get(model_name, {})
    model_params = {k: v() if callable(v) else v for k, v in model_params.items()}
    model = model_function(**model_params)

    results[model_name][dataset_name] = train_and_evaluate(dataset_name, model, X_labeled, y_labeled, X_unlabeled, X_test, y_test)    
    print(f"{dataset_name} Results:", results[model_name][dataset_name])
    print()
    save_results(results[model_name][dataset_name], model_name, dataset_name, model_params, labeled_size)


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
            'LapSVM' : LapSVM,
    #           'TemporalEnsembling' : TemporalEnsembling, 
    #           'MeanTeacher' : MeanTeacher,
    #           'VAT' : VAT,
            'TSVM' : TSVM,
            'LabelPropagation' : LabelPropagation
            }

    models_hyperparameters = {
                'Tri_Training' : {
                    'base_estimator': random_forest_factory,
                    'base_estimator_2': linear_svc_factory,
                    'base_estimator_3': gaussianNB_factory
                                    }
                ,'Assemble' : {
                    'base_estimator' : random_forest_factory
                }
                ,'SemiBoost' : {
                    'base_estimator' : random_forest_factory
                }
                ,'LapSVM' : {                  
                }
                ,'VAT' : {
                }
                ,'TSVM' : {
                }
                ,'LabelPropagation' : {
                }
                ,'TemporalEnsembling' : {
                }
                ,'MeanTeacher' : {
                    
                }
                }
    
    results = {
        'Tri_Training' : {},
        'Assemble' : {},
        'SemiBoost' : {},
        'LapSVM' : {},
        'VAT' : {},
        'TSVM' : {},
        'LabelPropagation' : {},
        'TemporalEnsembling' : {},
        'MeanTeacher' : {}
    }

    labeled_size = 0.3

    for model_name, model_function in models.items():
        for dataset_name, preprocess_function in datasets.items():
            train_model(model_name, model_function, dataset_name, preprocess_function, labeled_size, models_hyperparameters, results)


