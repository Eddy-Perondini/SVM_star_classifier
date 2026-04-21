import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd 
import kagglehub

#Carregando a Base de Dados 

path = kagglehub.dataset_download("deepu1109/star-dataset")

dados = pd.read_csv(f'{path}/6 class csv.csv')

#pré-processamento dos dados categóricos 

SColor_encoder = LabelEncoder()
SSpectral_encoder = LabelEncoder()

dados['Star color'] = SColor_encoder.fit_transform(dados['Star color'])
dados['Spectral Class'] = SSpectral_encoder.fit_transform(dados['Spectral Class'])

#Splitando o dataset 

X = dados.drop('Star type', axis = 1)
y = dados['Star type']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 67)

#Normalizando os dados 

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

SVM_classifier = SVC(kernel = 'rbf', random_state = 67, decision_function_shape= 'ovo')
SVM_classifier.fit(X_train_norm, y_train)

y_pred = SVM_classifier.predict(X_test_norm)

#Criando a visualização das barreiras de decisão da SVM

f1 = 'Temperature (K)'
f2 = 'Absolute magnitude(Mv)'

x_min, x_max = X[f1].min()-1000, X[f1].max()+1000
y_min, y_max = X[f2].min()-1, X[f2].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.zeros((xx.ravel().shape[0], X.shape[1]))

grid_df = pd.DataFrame(grid, columns=X.columns)

for col in X.columns:
    grid_df[col] = X[col].mean()

grid_df[f1] = xx.ravel()
grid_df[f2] = yy.ravel()

grid_norm = scaler.transform(grid_df)

Z = SVM_classifier.predict(grid_norm)
Z = Z.reshape(xx.shape)

print("Classification Report: \n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

#Plotando a matriz de confusão 

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (4,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')

#Plotando as barreiras de decisão da SVM

plt.figure(figsize=(4,4))

plt.contourf(xx, yy, Z, alpha=0.25, cmap='viridis')

plt.scatter(
    X[f1],
    X[f2],
    c=y,
    cmap='viridis',
    edgecolor='k'
)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.xlabel(f1)
plt.ylabel(f2)

plt.title("Barreiras de Decisão (SVM)")

plt.show()




