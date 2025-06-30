from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score
import pandas as pd
import numpy as np
from itertools import product

wdbc_db = pd.read_csv("database/wdbc.data", header=None)

# Preprocess the Dataset
wdbc_db.drop(columns=[0], inplace=True) # Remove the first column (ID)
wdbc_db.columns = [
    "diagnosis", 
    "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
]

# Split the dataset into features and target variable
X = wdbc_db.iloc[:, 1:].values
y = wdbc_db["diagnosis"]

# Split the dataset into training and testing sets into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Setting up Stratified K-Fold CV and Metrics
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

precision_scores = []
f1_scores = [] 
recall_scores = []

# Train and evaluate the Logistic Regression Classifier without Hyperparameter Tuning
for train_index, test_index in skf.split(X_train, y_train):
    features_train, features_test = X_train[train_index], X_train[test_index]
    labels_train, labels_test = y_train.iloc[train_index], y_train.iloc[test_index]
    
    model = LogisticRegression(random_state=42)
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)

    precision_scores.append(precision_score(labels_test, labels_pred, pos_label='M'))
    f1_scores.append(f1_score(labels_test, labels_pred, pos_label='M'))
    recall_scores.append(recall_score(labels_test, labels_pred, pos_label='M'))

print("Decision Tree Classifier Metrics on WDBC Dataset without Hyperparameter Tuning:") 
print("Average Precision:", np.mean(precision_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Average Recall Score:", np.mean(recall_scores))

# Train and evaluate the Logistic Regression Classifier with Hyperparameter Tuning
params = {
    'penalty': ['l1', 'l2', None],
    'C': [0.1, 0.2, 0.5, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
}

results = []

for penalty, C, solver in product(params['penalty'], params['C'], params['solver']):
    prcs, f1s, recs = [], [], []
    if (penalty == 'l1' and solver == 'lbfgs') or (penalty is None and solver == 'liblinear'):
        continue

    for train_index, test_index in skf.split(X_train, y_train):
        features_train, features_test = X_train[train_index], X_train[test_index]
        labels_train, labels_test = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42)
        model.fit(features_train, labels_train)
        labels_pred = model.predict(features_test)

        prcs.append(precision_score(labels_test, labels_pred, pos_label='M'))
        f1s.append(f1_score(labels_test, labels_pred, pos_label='M'))
        recs.append(recall_score(labels_test, labels_pred, pos_label='M'))
    
    results.append({
        'penalty': (penalty, C, solver),
        'precision': np.mean(prcs),
        'f1_score': np.mean(f1s),
        'recall_score': np.mean(recs)
    })

# Find best by each metric 
best_f1 = max(results, key=lambda x: x['f1_score'])
best_prc = max(results, key=lambda x: x['precision'])
best_rec = max(results, key=lambda x: x['recall_score'])
best_overall = max(results, key=lambda x: (x['f1_score'], x['precision'], x['recall_score']))

print("\nBest Hyperparameters by F1 Score:", best_f1['penalty'], "F1 Score:", best_f1['f1_score'])
print("Best Hyperparameters by Precision:", best_prc['penalty'], "Precision:", best_prc['precision'])
print("Best Hyperparameters by Recall Score:", best_rec['penalty'], "Recall Score:", best_rec['recall_score'])
print("Best Overall Hyperparameters:", best_overall['penalty'], "with F1 Score:", best_overall['f1_score'], 
      "Precision:", best_overall['precision'], "Recall Score:", best_overall['recall_score'])

# Evaluate the best model on the test set
best_model = LogisticRegression(penalty=best_overall['penalty'][0],
                                C=best_overall['penalty'][1], 
                                solver=best_overall['penalty'][2],
                                max_iter=best_overall['penalty'][3], 
                                random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nFinal Test Evaluation on Test Set with Best Overall Hyperparameters:")
print("Test Precision:", precision_score(y_test, y_pred, pos_label='M'))
print("Test F1 Score:", f1_score(y_test, y_pred, pos_label='M'))
print("Test Recall Score:", recall_score(y_test, y_pred, pos_label='M'))

