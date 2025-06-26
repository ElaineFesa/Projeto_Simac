
import cv2
import mediapipe as mp
import numpy as np
import joblib

modelo = joblib.load('modelo_letras_libras.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def extrair_vetor_caracteristicas(landmarks):
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            vetor = extrair_vetor_caracteristicas(hand_landmarks.landmark)
            predicao = modelo.predict([vetor])[0]

            cv2.putText(frame, f'Letra: {predicao}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Reconhecimento de Letras - Libras", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

