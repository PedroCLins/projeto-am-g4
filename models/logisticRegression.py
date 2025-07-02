from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score
import pandas as pd
import numpy as np
from itertools import product

def tuned_logistic_regression(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    # Setting up Stratified K-Fold CV, Metrics and Hyperparameters
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics_scores = []
    params = {
        'penalty': ['l1', 'l2', None],
        'C': [0.1, 0.2, 0.5, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
    }

    # Iterate through all combinations of hyperparameters to find the best model
    for penalty, C, solver in product(params['penalty'], params['C'], params['solver']):
        if (penalty == 'l1' and solver == 'lbfgs') or (penalty is None and solver == 'liblinear'):
            continue

        precision_scores, f1_scores, recall_scores = [], [], []

        for train_index, test_index in skf.split(X_train, y_train):
            features_train, features_test = X_train[train_index], X_train[test_index]
            labels_train, labels_test = y_train[train_index], y_train[test_index]
            model = LogisticRegression(
                penalty=penalty, 
                C=C, 
                solver=solver, 
                random_state=42
            )
            model.fit(features_train, labels_train)
            labels_pred = model.predict(features_test)

            precision_scores.append(precision_score(labels_test, labels_pred, pos_label=1))
            f1_scores.append(f1_score(labels_test, labels_pred, pos_label=1))
            recall_scores.append(recall_score(labels_test, labels_pred, pos_label=1))

        metrics_scores.append({
            'params': (penalty, C, solver),
            'precision': np.mean(precision_scores),
            'recall_score': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores),
        })

    # Best Hyperparameters Combination by Each Metric on Average
    print("LR Best Hyperparameters Combinations by Each Metric:")
    for metric in ['precision', 'recall_score', 'f1_score']:
        best_params = max(metrics_scores, key=lambda x: x[metric])
        print(f"{metric.capitalize()}: {best_params['params']} with score {best_params[metric]:.4f}")

    # Best Hyperparameters Combination Overall
    best_overall_params = max(metrics_scores, key=lambda x: (x['precision'] + x['f1_score'] + x['recall_score']) / 3)
    best_model = LogisticRegression(penalty=best_overall_params['params'][0],
                                    C=best_overall_params['params'][1],
                                    solver=best_overall_params['params'][2],
                                    random_state=42)

    return best_model, best_overall_params
