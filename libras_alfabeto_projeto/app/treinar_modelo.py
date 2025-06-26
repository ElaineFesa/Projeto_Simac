
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('dados/letras_libras.csv')
X = df.drop('letra', axis=1)
y = df['letra']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))

joblib.dump(modelo, 'modelo_letras_libras.pkl')
print("Modelo salvo como modelo_letras_libras.pkl")
