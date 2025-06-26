
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

os.makedirs('dados', exist_ok=True)
arquivo_csv = 'dados/letras_libras.csv'
if not os.path.exists(arquivo_csv):
    with open(arquivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['letra'] + [f'{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']])

def extrair_vetor_caracteristicas(landmarks):
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()

cap = cv2.VideoCapture(0)
print("Mostre a letra com a mão e pressione a tecla correspondente. ESC para sair.")

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

            cv2.putText(frame, "Pressione a letra correspondente", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Coletor de Letras Libras", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break  # ESC

    elif results.multi_hand_landmarks and 65 <= key <= 90 or 97 <= key <= 122:
        letra = chr(key).upper()
        with open(arquivo_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([letra] + vetor.tolist())
        print(f"Letra '{letra}' salva.")

cap.release()
cv2.destroyAllWindows()
