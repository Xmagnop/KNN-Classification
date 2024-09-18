import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#Plota os números juntos com as Labels de resultado real e previsto pelo KNN utilizando matplotlib
def plot_digits(images, labels, predictions=None):
    plt.figure(figsize=(10, 5))
    for index, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(2, 5, index + 1)
        plt.imshow(image.reshape(8, 8), cmap='gray')
        pred = predictions[index] if predictions is not None else ""
        label_str = 'Par' if label == 0 else 'Ímpar'
        pred_str = 'Par' if pred == 0 else 'Ímpar' if pred != "" else ""
        plt.title(f'True: {label_str}\nPred: {pred_str}')
        plt.axis('off')
    plt.show()

#Faz o KNN utilizando a técnica PCA
def PCA_KNN(digits):
    X = digits.data  #Características de entrada
    y = digits.target  #Label (saída)

    #Classifica os valores de saída com Par (0) ou ímpar (1)
    y = np.array([0 if label % 2 == 0 else 1 for label in y])

    #Divisão da base de dados em train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Normalização da base de dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Aplica PCA para redução de dimensionalidade
    n_components = 30
    pca = PCA(n_components=n_components)
    #Aplica tanto no conjunto train quanto test
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    #Criar KNN
    k = 3  #Define número de vizinhos como 3
    #Utiliza KNN do sklearn
    knn = KNeighborsClassifier(n_neighbors=k)

    #Treina o KNN com o conjunto Train após o PCA
    knn.fit(X_train_pca, y_train)

    #Faz as previsões
    y_pred = knn.predict(X_test_pca)

    #Avalia o accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy with PCA (Par/Ímpar): {accuracy * 100:.2f}%')

    #Mostrar o resultado
    print('\nClassification Report (Par/Ímpar):')
    print(classification_report(y_test, y_pred))

    #Mostra 10 exemplos de predição
    sample_images = X_test[:10]
    sample_labels = y_test[:10]
    sample_predictions = y_pred[:10]
    plot_digits(sample_images, sample_labels, sample_predictions)

#Faz o KNN utilizando a técnica LDA
def LDA_KNN(digits):
    X = digits.data  #Características de entrada
    y = digits.target  #Label (saída)

    #Classifica os valores de saída com Par (0) ou ímpar (1)
    y = np.array([0 if label % 2 == 0 else 1 for label in y])

    #Divisão da base de dados em train e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Normalização da base de dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Aplica LDA para redução de dimensionalidade
    n_components = 1  #Definir o número de componentes principais como 1 (para 2 classes)
    lda = LDA(n_components=n_components)
    #Aplica tanto no conjunto train quanto test
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    #Balancear dados com SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_lda, y_train)

    #Criar KNN com 20 vizinhos e métrica Manhattan
    k = 20  #Número de vizinhos
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')

    #Treina o KNN com o conjunto Train após o LDA
    knn.fit(X_train_res, y_train_res)

    #Faz as previsões
    y_pred = knn.predict(X_test_lda)

    #Avalia o accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy with LDA (Par/Ímpar): {accuracy * 100:.2f}%')

    #Cross-validation
    scores = cross_val_score(knn, X_train_lda, y_train, cv=5)
    print(f'Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%')

    #Mostrar o resultado
    print('\nClassification Report (Par/Ímpar):')
    print(classification_report(y_test, y_pred))

    #Mostra 10 exemplos de predição
    sample_images = X_test[:10]
    sample_labels = y_test[:10]
    sample_predictions = y_pred[:10]
    plot_digits(sample_images, sample_labels, sample_predictions)

def main():

    #Carregar o conjunto de dados MNIST
    digits = datasets.load_digits()
    #Executar funções criadas
    PCA_KNN(digits)
    LDA_KNN(digits)

    return 0

if __name__ == "__main__":
    SystemExit(main())
