from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score
import pandas as pd
import numpy as np
from itertools import product

def tuned_decision_tree(X_train:np.ndarray, y_train:pd.Series) -> DecisionTreeClassifier:
    # Setting up Stratified K-Fold CV, Metrics and Hyperparameters
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics_scores = []
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
    }

    # Iterate through all combinations of hyperparameters to find the best model
    for criterion, max_depth, min_samples_split in product(params['criterion'], params['max_depth'], params['min_samples_split']):
        precision_scores, f1_scores, recall_scores = [], [], []

        for train_index, test_index in skf.split(X_train, y_train):
            features_train, features_test = X_train[train_index], X_train[test_index]
            labels_train, labels_test = y_train[train_index], y_train[test_index]
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(features_train, labels_train)
            labels_pred = model.predict(features_test)

            precision_scores.append(precision_score(labels_test, labels_pred, pos_label=1))
            f1_scores.append(f1_score(labels_test, labels_pred, pos_label=1))
            recall_scores.append(recall_score(labels_test, labels_pred, pos_label=1))
        
        metrics_scores.append({
            'params': (criterion, max_depth, min_samples_split),
            'precision': np.mean(precision_scores),
            'recall_score': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores),
        })

    # Best Hyperparameters Combination by Each Metric on Average
    print("DT Best Hyperparameters Combinations by Each Metric:")
    for metric in ['precision', 'recall_score', 'f1_score']:
        best_params = max(metrics_scores, key=lambda x: x[metric])
        print(f"{metric.capitalize()}: {best_params['params']} with score {best_params[metric]:.4f}")

    # Best Hyperparameters Combination Overall
    best_overall_params = max(metrics_scores, key=lambda x: (x['precision'] + x['f1_score'] + x['recall_score']) / 3)
    best_model = DecisionTreeClassifier(criterion=best_overall_params['params'][0],
                                        max_depth=best_overall_params['params'][1],
                                        min_samples_split=best_overall_params['params'][2],
                                        random_state=42)

    return best_model, best_overall_params
