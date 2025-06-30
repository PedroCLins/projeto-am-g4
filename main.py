from models.decisionTree import tuned_decision_tree
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# Defining the Decision Tree Classifier with Hyperparameter Tuning, Cross-Validator and Metrics
tree_model, y_pred = tuned_decision_tree(X_train, y_train)
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
scorers = {'precision': make_scorer(precision_score, pos_label='M'),
           'recall': make_scorer(recall_score, pos_label='M'),
           'f1': make_scorer(f1_score, pos_label='M')}

for score in scorers:
    train_sizes, train_scores, test_scores = learning_curve(
        tree_model, X, y, cv=skf, 
        train_sizes=np.arange(0.05, 1.01, 0.05), 
        scoring=scorers[score], 
        random_state=0)
    
    # Converting train_sizes to percentage
    train_sizes_percent = train_sizes / X.shape[0] * 100
    
    fig, ax = plt.subplots()
    # Plot using percentage as x-axis
    ax.plot(train_sizes_percent, np.mean(train_scores, axis=1), 'o-', label="Training score")
    ax.plot(train_sizes_percent, np.mean(test_scores, axis=1), 'o-', label="Cross-validation score")
    ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel(f"{score.capitalize()} Score")
    ax.set_title(f"Learning Curve ({score.capitalize()})")
    ax.legend(loc="best")
    # Set x-ticks to every 5%
    ax.set_xticks(np.arange(5, 101, 5))
    ax.grid(True)
    plt.tight_layout()
    plt.show()

