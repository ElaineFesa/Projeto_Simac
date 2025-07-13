import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# ================= CONFIGURAÇÕES =================
SEQUENCE_LENGTH = 30                # Número fixo de frames por gesto
MIN_SAMPLES_PER_CLASS = 5           # Mínimo de amostras por gesto
DATA_DIR = 'dados_gestos'           # Pasta com os dados
MODEL_NAME = 'modelo_gestos_libras.h5'  # Nome do modelo de saída

# ============== FUNÇÕES AUXILIARES ==============
def verificar_dados(diretorio, min_amostras):
    """Verifica a quantidade de amostras por gesto e retorna os válidos"""
    contagem = {}
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".pkl"):
            with open(os.path.join(diretorio, arquivo), 'rb') as f:
                try:
                    contador = 0
                    while True:
                        pickle.load(f)
                        contador += 1
                except EOFError:
                    pass
            nome_gesto = arquivo.split('.')[0]
            contagem[nome_gesto] = contador
    
    print("\n=== ANÁLISE DOS DADOS ===")
    for gesto, qtd in contagem.items():
        status = "✅" if qtd >= min_amostras else f"❌ (adicione {min_amostras - qtd} amostras)"
        print(f"{gesto}: {qtd} amostras {status}")
    
    gestos_validos = [g for g, q in contagem.items() if q >= min_amostras]
    
    if not gestos_validos:
        print("\nERRO CRÍTICO: Nenhum gesto com amostras suficientes")
        print(f"Mínimo necessário: {min_amostras} amostras por gesto")
        return None
    
    print(f"\nGestos que serão usados: {', '.join(gestos_validos)}")
    return gestos_validos

def carregar_dados(diretorio, gestos_validos):
    """Carrega e retorna os dados filtrados"""
    X, y = [], []
    
    for arquivo in os.listdir(diretorio):
        if not arquivo.endswith(".pkl"):
            continue
            
        rotulo = arquivo.split(".")[0]
        if rotulo not in gestos_validos:
            continue
            
        caminho = os.path.join(diretorio, arquivo)
        
        with open(caminho, 'rb') as f:
            while True:
                try:
                    gestos = pickle.load(f)
                    for gesto in gestos:
                        if gesto.shape[0] == SEQUENCE_LENGTH:
                            X.append(gesto)
                            y.append(rotulo)
                except EOFError:
                    break
                    
    return np.array(X), np.array(y)

# ============ VALIDAÇÃO INICIAL ============
print("\n=== INÍCIO DO TREINAMENTO ===")
gestos_validos = verificar_dados(DATA_DIR, MIN_SAMPLES_PER_CLASS)
if not gestos_validos:
    exit(1)

# ============= CARREGAR DADOS ==============
print("\nCarregando dados...")
X, y = carregar_dados(DATA_DIR, gestos_validos)

if len(X) == 0:
    print("ERRO: Nenhum dado válido encontrado após filtragem")
    print("Verifique se os arquivos .pkl contêm arrays no formato correto")
    exit(1)

print(f"\nDados carregados:")
print(f"- Total de amostras: {len(X)}")
print(f"- Formatos dos dados: {X.shape}")

# ============ PRÉ-PROCESSAMENTO ============
# Codifica os rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# ============== MODELO LSTM ===============
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 126)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============= TREINAMENTO ================
print("\nIniciando treinamento...")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ============ SALVAR MODELO ===============
model.save(MODEL_NAME)
joblib.dump(le, "rotulador_gestos.pkl")

print("\n=== TREINAMENTO CONCLUÍDO ===")
print(f"Modelo salvo como: {MODEL_NAME}")
print(f"Gestos reconhecíveis: {list(le.classes_)}")
print(f"Acurácia final: {history.history['val_accuracy'][-1]:.2%}")