from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score
import pandas as pd
import numpy as np

def tuned_naive_bayes(X_train: np.ndarray, y_train: pd.Series) -> GaussianNB:
    # Setting up Stratified K-Fold CV, Metrics and Hyperparameters
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics_scores = []
    params = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }

    # Iterate through all combinations of hyperparameters to find the best model
    for var_smoothing in params['var_smoothing']:
        precision_scores, f1_scores, recall_scores = [], [], []

        for train_index, test_index in skf.split(X_train, y_train):
            features_train, features_test = X_train[train_index], X_train[test_index]
            labels_train, labels_test = y_train[train_index], y_train[test_index]
            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(features_train, labels_train)
            labels_pred = model.predict(features_test)

            precision_scores.append(precision_score(labels_test, labels_pred, pos_label=1))
            f1_scores.append(f1_score(labels_test, labels_pred, pos_label=1))
            recall_scores.append(recall_score(labels_test, labels_pred, pos_label=1))

        metrics_scores.append({
            'params': (var_smoothing,),
            'precision': np.mean(precision_scores),
            'recall_score': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores),
        })

    # Best Hyperparameters Combination by Each Metric on Average
    print("Best Hyperparameters Combinations by Each Metric:")
    for metric in ['precision', 'recall_score', 'f1_score']:
        best_params = max(metrics_scores, key=lambda x: x[metric])
        print(f"{metric.capitalize()}: {best_params['params']} with score {best_params[metric]:.4f}")

    # Best Hyperparameters Combination Overall
    best_overall_params = max(metrics_scores, key=lambda x: (x['precision'] + x['f1_score'] + x['recall_score']) / 3)
    best_model = GaussianNB(var_smoothing=best_overall_params['params'][0])

    return best_model, best_overall_params