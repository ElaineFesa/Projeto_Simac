import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import deque
import mediapipe as mp
from pathlib import Path

# Configurações
MODEL_PATH = Path('modelos/modelo_gestos.h5')
LABEL_PATH = Path('modelos/rotulador_gestos.pkl')
SEQUENCE_LENGTH = 30
MIN_CONFIDENCE = 0.7

# Verificação inicial
if not MODEL_PATH.exists() or not LABEL_PATH.exists():
    print("❌ Modelo não encontrado!")
    print("Execute primeiro o treinamento:")
    print("> python treinar_gestos.py")
    exit()

# Inicialização
model = load_model(MODEL_PATH)
le = joblib.load(LABEL_PATH)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

buffer = deque(maxlen=SEQUENCE_LENGTH)
cap = cv2.VideoCapture(0)

print("\n=== RECONHECIMENTO DE GESTOS ===")
print(f"Gestos carregados: {', '.join(le.classes_)}")
print("Pressione ESC para sair\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # Processa landmarks (2 mãos)
        landmarks = []
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
            landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        
        # Padronização
        landmarks = landmarks[:42]  # Máximo 2 mãos
        if len(landmarks) < 42:
            landmarks.extend([[0,0,0]] * (42 - len(landmarks)))
        
        buffer.append(np.array(landmarks).flatten())

        # Reconhecimento quando buffer cheio
        if len(buffer) == SEQUENCE_LENGTH:
            entrada = np.array(buffer).reshape(1, SEQUENCE_LENGTH, 126)
            preds = model.predict(entrada, verbose=0)[0]
            classe_idx = np.argmax(preds)
            confianca = preds[classe_idx]

            if confianca >= MIN_CONFIDENCE:
                gesto = le.classes_[classe_idx]
                cv2.putText(frame, f"{gesto} ({confianca:.0%})", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()