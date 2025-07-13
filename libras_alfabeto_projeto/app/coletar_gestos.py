import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from collections import deque
import time

# Configurações
FRAMES_POR_SEGUNDO = 30  # Taxa de amostragem
MIN_DURACAO = 1.0        # Duração mínima do gesto em segundos
MAX_DURACAO = 3.0        # Duração máxima do gesto em segundos

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def normalizar_vetor(landmarks):
    vetor = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    return vetor - vetor[0]

def redimensionar_gesto(gesto_original, novo_tamanho=30):
    """Redimensiona o gesto para ter um número fixo de frames"""
    from scipy.interpolate import interp1d
    
    tempo_original = np.linspace(0, 1, len(gesto_original))
    tempo_novo = np.linspace(0, 1, novo_tamanho)
    
    gesto_redimensionado = []
    for i in range(gesto_original.shape[1]):
        interp = interp1d(tempo_original, gesto_original[:, i], kind='linear')
        gesto_redimensionado.append(interp(tempo_novo))
    
    return np.column_stack(gesto_redimensionado)

print("\n=== COLETOR INTELIGENTE DE GESTOS ===")
print("Instruções:")
print("1. Digite o nome do gesto e pressione ENTER")
print("2. Pressione ESPAÇO para INICIAR a gravação")
print("3. Execute o gesto naturalmente")
print("4. Pressione ESPAÇO novamente para FINALIZAR")
print("Pressione ESC para sair\n")

while True:
    gesto_nome = input("Nome do gesto (deixe vazio para sair): ").strip().upper()
    if not gesto_nome:
        break
    
    buffer = []
    gravando = False
    tempo_inicio = 0
    
    print(f"\nPronto para gravar: {gesto_nome}")
    print("Pressione ESPAÇO para começar...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Processa detecção de mãos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Prepara vetor normalizado
            vetor_maos = []
            for hand_landmarks in results.multi_hand_landmarks:
                vetor_maos.append(normalizar_vetor(hand_landmarks.landmark).flatten())
            
            if len(vetor_maos) == 1:
                vetor_completo = np.concatenate([vetor_maos[0], np.zeros(63)])
            else:
                vetor_completo = np.concatenate(vetor_maos[:2])
            
            if gravando:
                buffer.append(vetor_completo)
                duracao = time.time() - tempo_inicio
                cv2.putText(frame, f"Gravando: {duracao:.1f}s", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Controles
        key = cv2.waitKey(1)
        if key == 32:  # ESPAÇO
            if not gravando:
                gravando = True
                tempo_inicio = time.time()
                buffer = []
                print("Gravação INICIADA - Execute o gesto...")
            else:
                gravando = False
                duracao = time.time() - tempo_inicio
                
                if duracao < MIN_DURACAO:
                    print(f"Gestos muito curtos! Mínimo: {MIN_DURACAO}s")
                    continue
                
                # Redimensiona para 30 frames mantendo proporção
                gesto_redimensionado = redimensionar_gesto(np.array(buffer))
                
                # Salva o gesto
                os.makedirs('dados_gestos', exist_ok=True)
                with open(f'dados_gestos/{gesto_nome}.pkl', 'ab') as f:
                    pickle.dump([gesto_redimensionado], f)
                
                print(f"✅ Gesto '{gesto_nome}' salvo! Duração: {duracao:.1f}s")
                break
                
        elif key == 27:  # ESC
            break
        
        cv2.imshow("Coletor de Gestos", frame)
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()