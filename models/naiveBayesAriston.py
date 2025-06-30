import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------
# Definição da Classe NaiveBayes (Tudo no mesmo arquivo)
# --------------------------------------------------------------------------
class NaiveBayes:
    """
    Implementação do classificador Gaussian Naive Bayes a partir do zero.
    """
    def fit(self, X, y):
        """
        Treina o modelo.

        Calcula a média, variância e probabilidade a priori para cada classe.
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Inicializa as estruturas para armazenar médias, variâncias e priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            # Filtra as amostras que pertencem à classe atual
            X_c = X[y == c]
            
            # Calcula e armazena a média e a variância para cada feature da classe
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            
            # Calcula e armazena a probabilidade a priori da classe
            # (frequência da classe no conjunto de treino)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Realiza a predição para um conjunto de dados X.
        """
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        """
        Prevê a classe para uma única amostra x.
        """
        posteriors = []

        # Calcula a probabilidade posterior para cada classe
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            # Calcula o log da verossimilhança (likelihood) usando a PDF Gaussiana
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Retorna a classe com a maior probabilidade posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Função de Densidade de Probabilidade (PDF) Gaussiana.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        # Adiciona uma pequena constante (epsilon) para evitar divisão por zero
        return numerator / (denominator + 1e-9)

# --------------------------------------------------------------------------
# Carregamento, Treinamento e Avaliação
# --------------------------------------------------------------------------

# --- Carregamento e Pré-processamento dos Dados ---
# Carregando o dataset a partir de uma URL para reprodutibilidade
# Fonte: UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
wdbc_db = pd.read_csv(url, header=None)

# Removendo a primeira coluna (ID) e definindo nomes das colunas
wdbc_db.drop(columns=[0], inplace=True)
wdbc_db.columns = [
    "diagnosis",
    "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
]

# Separando features (X) e alvo (y)
X = wdbc_db.iloc[:, 1:].values
y = wdbc_db["diagnosis"]

# Codificando o alvo (y) para valores numéricos: 'M' -> 1, 'B' -> 0
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Divisão em Treino e Teste (70/30) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# --- Instanciando os modelos ---
naive_sklearn = GaussianNB()
my_naive = NaiveBayes() # Sua implementação

# --- Treinando os modelos ---
naive_sklearn.fit(X_train, y_train)
my_naive.fit(X_train, y_train)

# --- Realizando as Predições ---
y_pred_naive_sklearn = naive_sklearn.predict(X_test)
y_pred_my_naive = my_naive.predict(X_test)

# --- Exibindo os Relatórios de Classificação ---
print("="*60)
print("  AVALIAÇÃO DOS MODELOS NAIVE BAYES NO DATASET WDBC (70/30)")
print("="*60)

print("\n--- Relatório de Classificação do Modelo GaussianNB (Scikit-learn) ---\n")
# Usamos le.classes_ para mostrar os nomes originais ('B', 'M') no relatório
print(classification_report(y_test, y_pred_naive_sklearn, target_names=le.classes_))

print("\n--- Relatório de Classificação do seu Modelo NaiveBayes (my_naive) ---\n")
print(classification_report(y_test, y_pred_my_naive, target_names=le.classes_))