import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

arquivo_csv = 'dados/letras_libras.csv'
if not os.path.exists(arquivo_csv):
    print("Arquivo de dados não encontrado.")
    exit()

df = pd.read_csv(arquivo_csv)
df = df.groupby('letra').filter(lambda x: len(x) >= 10)
X = df.drop('letra', axis=1)
y = df['letra']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

joblib.dump(modelo, 'modelo_letras_libras.pkl')
print("Modelo salvo como modelo_letras_libras.pkl")