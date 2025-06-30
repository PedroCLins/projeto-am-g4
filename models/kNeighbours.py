import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Carregar dados
data = pd.read_csv('database/wdbc.data', header=None)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].map({'M': 1, 'B': 0}).values  # M=1, B=0

# Separar treino+validação e teste (30% do total)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



# Encontrar melhor k e melhor weights usando validação cruzada
k_range = range(1, 21)
weights_options = ['uniform', 'distance']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_score = -1
best_k = None
best_weights = None

for weights in weights_options:
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights=weights)
        scores = cross_val_score(knn, X_trainval, y_trainval, cv=cv, scoring='f1')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
            best_weights = weights

print(f"Melhor k encontrado: {best_k}")
print(f"Melhor weights encontrado: {best_weights}")

# Treinar com melhor k e melhor weights
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', weights=best_weights)
knn.fit(X_trainval, y_trainval)
y_pred = knn.predict(X_test)

# Avaliação
print("Relatório de classificação no conjunto de teste:")
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# Curvas de aprendizagem
train_sizes = np.arange(0.05, 1.0, 0.05)
precisions, recalls, f1s = [], [], []
for train_size in train_sizes:
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes * 100, precisions, label='Precision')
plt.plot(train_sizes * 100, recalls, label='Recall')
plt.plot(train_sizes * 100, f1s, label='F1-score')
plt.xlabel('Percentual do conjunto de treino (%)')
plt.ylabel('Métrica no conjunto de teste')
plt.title('Curvas de aprendizagem para k-NN (k=%d)' % best_k)
plt.legend()
plt.grid(True)
plt.show()