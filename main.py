from models.logisticRegression import tuned_logistic_regression
from models.decisionTree import tuned_decision_tree
from models.kNeighbours import tuned_k_neighbors
from models.naiveBayes import tuned_naive_bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wdbc_db = pd.read_csv("database/wdbc.data", header=None)

# Preprocessing the Dataset
wdbc_db.drop(columns=[0], inplace=True) # Remove the first column (ID)
wdbc_db.columns = [
    "diagnosis", 
    "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
]

# Split the dataset into features and target variable
label_encoder = LabelEncoder()
X = wdbc_db.iloc[:, 1:].values
y = label_encoder.fit_transform(wdbc_db["diagnosis"])  # Encode 'M' as 1 and 'B' as 0

# Split the dataset into training and testing sets into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Defining the Classifiers with Hyperparameter Tuning, Cross-Validator and Metrics
logreg_model = tuned_logistic_regression(X_train, y_train)
tree_model = tuned_decision_tree(X_train, y_train)
knn_model = tuned_k_neighbors(X_train, y_train)
nb_model = tuned_naive_bayes(X_train, y_train)

skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
scorers = {'Precision': make_scorer(precision_score, pos_label=1),
           'Recall': make_scorer(recall_score, pos_label=1),
           'F1': make_scorer(f1_score, pos_label=1)}
models = {
    'Logistic Regression': logreg_model,
    'Decision Tree': tree_model,
    'K-Nearest Neighbors': knn_model,
    'Gaussian Naive Bayes': nb_model
}

train_sizes = np.arange(0.05, 1.0, 0.05)
train_sizes_percent = (train_sizes * 100).astype(int)

for scorer_name, scorer in scorers.items():
    fig, ax = plt.subplots()
    for model_name, model in models.items():
        scores = []
        for train_size in train_sizes:
            X_train_curve, X_test_curve, y_train_curve, y_test_curve = train_test_split(
                X, y, train_size=train_size, random_state=42, stratify=y
            )
            model.fit(X_train_curve, y_train_curve)
            y_pred_curve = model.predict(X_test_curve)
            scores.append(scorer._score_func(y_test_curve, y_pred_curve, pos_label=1))
        ax.plot(train_sizes_percent, scores, marker='o', label=model_name)

    ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel(f"{scorer_name} Score")
    ax.set_title(f"Learning Curve Comparison ({scorer_name})")
    ax.legend(loc="best")
    ax.set_xticks(np.arange(5, 100, 5))
    ax.grid(True)
    plt.tight_layout()
    plt.show()