import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import deque
import time

# Configurações
SEQUENCE_LENGTH = 30
MIN_CONFIDENCE = 0.85  # Confiança mínima para considerar reconhecimento
HISTORY_SIZE = 5       # Tamanho do histórico para suavização

# Carrega modelo e codificador
model = load_model("modelo_gestos_libras.h5")
le = joblib.load("rotulador_gestos.pkl")

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)
mp_drawing = mp.solutions.drawing_utils

# Inicializa câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

# Variáveis para reconhecimento
buffer = deque(maxlen=SEQUENCE_LENGTH)
historico = deque(maxlen=HISTORY_SIZE)
ultimo_gesto = None
ultimo_tempo = 0

def normalizar_vetor(landmarks):
    vetor = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return vetor - vetor[0]

print("\n=== RECONHECIMENTO DE GESTOS ===")
print("Mostre o gesto com uma ou duas mãos")
print("Pressione ESC para sair\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Processa detecção de mãos
    if results.multi_hand_landmarks:
        vetor_maos = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            vetor_maos.append(normalizar_vetor(hand_landmarks.landmark).flatten())
        
        # Preenche com zeros se apenas uma mão for detectada
        if len(vetor_maos) == 1:
            vetor_completo = np.concatenate([vetor_maos[0], np.zeros(63)])
        else:
            vetor_completo = np.concatenate(vetor_maos[:2])
        
        buffer.append(vetor_completo)
        
        # Quando temos uma sequência completa
        if len(buffer) == SEQUENCE_LENGTH:
            entrada = np.expand_dims(np.array(buffer), axis=0)
            preds = model.predict(entrada, verbose=0)[0]
            classe_idx = np.argmax(preds)
            confianca = preds[classe_idx]
            
            if confianca >= MIN_CONFIDENCE:
                gesto_atual = le.classes_[classe_idx]
                historico.append(gesto_atual)
                
                # Suavização por voto majoritário
                contagem = {}
                for g in historico:
                    contagem[g] = contagem.get(g, 0) + 1
                gesto_final = max(contagem.items(), key=lambda x: x[1])[0]
                
                # Evita repetições rápidas
                if gesto_final != ultimo_gesto or (time.time() - ultimo_tempo) > 1:
                    print(f"Gesto reconhecido: {gesto_final} ({confianca:.2%})")
                    ultimo_gesto = gesto_final
                    ultimo_tempo = time.time()
                
                # Exibe na tela
                cv2.putText(frame, f"{gesto_final} ({confianca:.0%})", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    else:
        buffer.clear()
    
    # Exibe informações
    cv2.putText(frame, f"Buffer: {len(buffer)}/{SEQUENCE_LENGTH}", 
               (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    cv2.imshow("Reconhecimento de Gestos", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()