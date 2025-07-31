import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from collections import deque
from pathlib import Path
from datetime import timedelta

# === CONFIGURAÇÕES ===
VIDEO_PATH = r'C:\Temp\Projeto_Simac\entrada\video_libras.mp4'
OUTPUT_DIR = Path('saida')
OUTPUT_DIR.mkdir(exist_ok=True)
LEGEND_FILE = OUTPUT_DIR / 'legenda.txt'
OUTPUT_VIDEO = OUTPUT_DIR / 'video_com_legenda.mp4'
MODEL_PATH = Path('modelos/modelo_gestos.h5')
LABEL_PATH = Path('modelos/rotulador_gestos.pkl')
SEQUENCE_LENGTH = 15
MIN_CONFIDENCE = 0.5


# === CARREGAMENTO DE MODELOS ===
model = load_model(MODEL_PATH)
le = joblib.load(LABEL_PATH)

# === MEDIAPIPE ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# === FUNÇÃO DE RECONHECIMENTO ===
def reconhecer_gesto(buffer):
    entrada = np.array(buffer).reshape(1, SEQUENCE_LENGTH, 126)
    preds = model.predict(entrada, verbose=0)[0]
    idx = np.argmax(preds)
    confianca = preds[idx]
    if confianca >= MIN_CONFIDENCE:
        return le.classes_[idx], confianca
    return None, None

# === PROCESSAMENTO DO VÍDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Erro ao abrir o vídeo.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (largura, altura))

buffer = deque(maxlen=SEQUENCE_LENGTH)
legendas = []
frame_idx = 0
gestos_detectados = 0

print("📹 Processando vídeo...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks = []
    if results.multi_hand_landmarks:
        print(f"[Frame {frame_idx}] Mãos detectadas")

        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
            landmarks.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])

        landmarks = landmarks[:42]
        if len(landmarks) < 42:
            landmarks.extend([[0, 0, 0]] * (42 - len(landmarks)))

        buffer.append(np.array(landmarks).flatten())

        if len(buffer) == SEQUENCE_LENGTH and gestos_detectados == 0:
            print(f"[Reconhecimento] Buffer completo no frame {frame_idx}")
            classe, confianca = reconhecer_gesto(buffer)
            timestamp = str(timedelta(seconds=frame_idx / fps))

            if classe:
                legenda = f"[{timestamp}] {classe} ({confianca:.0%})"
                legendas.append(legenda)
                gestos_detectados += 1

                cv2.putText(frame, f'{classe} ({confianca:.0%})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Reconhecendo...", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    else:
        cv2.putText(frame, "Mao nao detectada", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# === SALVAR LEGENDA ===
with open(LEGEND_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(legendas))

print(f"\n✅ Processamento finalizado!")
print(f"📄 Legenda salva em: {LEGEND_FILE}")
print(f"🎞️ Vídeo gerado: {OUTPUT_VIDEO}")
print(f"✋ Total de gestos detectados: {gestos_detectados}")
