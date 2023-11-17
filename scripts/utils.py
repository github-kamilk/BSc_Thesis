from datetime import datetime
from sklearn.model_selection import train_test_split
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB


def split_labeled_unlabeled(X, y, labeled_size):
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, train_size=labeled_size, random_state=42, stratify=y)
    return X_labeled, y_labeled, X_unlabeled


def save_results(results, model_name, dataset_name, model_params, labeled_size):
    data_to_save = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": model_name,
        'dataset': dataset_name,
        'labeled_size': labeled_size,
        "hyperparameters": str(model_params),
        "results": results
    }
    with open(f"experiments_results_base_parameters/{model_name}_{dataset_name}.json", 'a') as f, open('experiments_results_base_parameters/all_results.json', 'a') as g:
        json.dump(data_to_save, f, indent=4)
        json.dump(data_to_save, g, indent=4)

def knn_factory():
    return KNeighborsClassifier(n_neighbors=7)

def random_forest_factory():
    return RandomForestClassifier()

def linear_svc_factory():
    return LinearSVC(max_iter=10**4)

def gaussianNB_factory():
    return GaussianNB()