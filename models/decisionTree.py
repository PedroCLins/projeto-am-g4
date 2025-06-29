from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
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

accuracy_scores = []
f1_scores = [] 
recall_scores = []

# Train and evaluate the Decision Tree Classifier without Hyperparameter Tuning
for train_index, test_index in skf.split(X_train, y_train):
    features_train, features_test = X_train[train_index], X_train[test_index]
    labels_train, labels_test = y_train.iloc[train_index], y_train.iloc[test_index]
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(features_train, labels_train)
    labels_pred = model.predict(features_test)

    accuracy_scores.append(accuracy_score(labels_test, labels_pred))
    f1_scores.append(f1_score(labels_test, labels_pred, pos_label='M'))
    recall_scores.append(recall_score(labels_test, labels_pred, pos_label='M'))

print("Decision Tree Classifier Metrics on WDBC Dataset without Hyperparameter Tuning:") 
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Average Recall Score:", np.mean(recall_scores))

# Train and evaluate the Decision Tree Classifier with Hyperparameter Tuning
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []

for criterion, max_depth, min_samples_split in product(params['criterion'], params['max_depth'], params['min_samples_split']):
    accs, f1s, recs = [], [], []
    for train_index, test_index in skf.split(X_train, y_train):
        features_train, features_test = X_train[train_index], X_train[test_index]
        labels_train, labels_test = y_train.iloc[train_index], y_train.iloc[test_index]
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(features_train, labels_train)
        labels_pred = model.predict(features_test)

        accs.append(accuracy_score(labels_test, labels_pred))
        f1s.append(f1_score(labels_test, labels_pred, pos_label='M'))
        recs.append(recall_score(labels_test, labels_pred, pos_label='M'))
    results.append({
        'params': (criterion, max_depth, min_samples_split),
        'accuracy': np.mean(accs),
        'f1_score': np.mean(f1s),
        'recall_score': np.mean(recs)
    })

# Find best by each metric
best_f1 = max(results, key=lambda x: x['f1_score'])
best_acc = max(results, key=lambda x: x['accuracy'])
best_rec = max(results, key=lambda x: x['recall_score'])
best_overall = max(results, key=lambda x: (x['accuracy'] + x['f1_score'] + x['recall_score']) / 3)

print("\nBest Hyperparameters based on F1 Score:", best_f1['params'], "with F1 Score:", best_f1['f1_score'])
print("Best Hyperparameters based on Accuracy:", best_acc['params'], "with Accuracy:", best_acc['accuracy'])
print("Best Hyperparameters based on Recall Score:", best_rec['params'], "with Recall Score:", best_rec['recall_score'])
print("\nBest Overall Hyperparameters:", best_overall['params'], "with Average Score:", 
      (best_overall['accuracy'] + best_overall['f1_score'] + best_overall['recall_score']) / 3)

# Evaluate the best model on the test set
best_model = DecisionTreeClassifier(criterion=best_overall['params'][0],
                                    max_depth=best_overall['params'][1],
                                    min_samples_split=best_overall['params'][2],
                                    random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nFinal Evaluation on Test Set with Best Overall Hyperparameters:")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Score:", f1_score(y_test, y_pred, pos_label='M'))
print("Test Recall Score:", recall_score(y_test, y_pred, pos_label='M'))