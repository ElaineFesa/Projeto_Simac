import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class TelaVogais(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.letras_vogais = ['A', 'E', 'I', 'O', 'U']
        self.letra_atual = 0
        self.acertos = 0
        self.setup_ui()
        self.setup_camera()
        self.setup_mediapipe()
    
    def setup_ui(self):
        self.setStyleSheet("background-color: #2c3e50; color: white;")
        
        self.layout = QVBoxLayout()
        
        # Cabeçalho
        self.lbl_instrucao = QLabel(f"Faça o gesto da letra: {self.letras_vogais[self.letra_atual]}")
        self.lbl_instrucao.setStyleSheet("font-size: 24px; color: #f1c40f;")
        
        # Visualização da câmera
        self.lbl_camera = QLabel()
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        
        # Feedback
        self.lbl_feedback = QLabel("Mostre sua mão para a câmera")
        self.lbl_feedback.setStyleSheet("font-size: 20px; color: #3498db;")
        
        # Botão Voltar
        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(self.voltar)
        
        self.layout.addWidget(self.lbl_instrucao)
        self.layout.addWidget(self.lbl_camera)
        self.layout.addWidget(self.lbl_feedback)
        self.layout.addWidget(btn_voltar)
        
        self.setLayout(self.layout)
    
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.timer.start(30)  # 30 FPS
    
    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def atualizar_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Aqui você implementaria a lógica de reconhecimento
                # Esta é uma simplificação - você precisará treinar seu modelo
                letra_reconhecida = self.reconhecer_gesto(hand_landmarks)
                
                if letra_reconhecida == self.letras_vogais[self.letra_atual]:
                    self.lbl_feedback.setText(f"Correto! Você fez a letra {letra_reconhecida}")
                    self.lbl_feedback.setStyleSheet("color: #2ecc71; font-size: 20px;")
                    self.acertos += 1
                    self.proxima_letra()
                else:
                    self.lbl_feedback.setText(f"Tente novamente. Você fez: {letra_reconhecida}")
                    self.lbl_feedback.setStyleSheet("color: #e74c3c; font-size: 20px;")
        
        # Mostra o frame
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.lbl_camera.setPixmap(QPixmap.fromImage(p))
    
    def reconhecer_gesto(self, landmarks):
        # SIMULAÇÃO - Substitua por seu modelo real de reconhecimento
        # Aqui você deve implementar seu algoritmo ou modelo ML
        # Esta é apenas uma demonstração que alterna entre vogais
        
        # Coordenadas normalizadas dos landmarks
        coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        
        # Exemplo simplificado (substitua por sua lógica real)
        gestos = {
            0: 'A',  # Mão aberta
            1: 'E',  # Mão semi-fechada
            2: 'I',  # Indicador para cima
            3: 'O',  # Mão em forma de O
            4: 'U'   # Indicador e médio para cima
        }
        
        # Simplesmente rotaciona entre as vogais (substitua por reconhecimento real)
        return gestos.get(self.letra_atual % 5, 'A')
    
    def proxima_letra(self):
        self.letra_atual += 1
        if self.letra_atual < len(self.letras_vogais):
            self.lbl_instrucao.setText(f"Faça o gesto da letra: {self.letras_vogais[self.letra_atual]}")
        else:
            self.lbl_instrucao.setText(f"Parabéns! Você acertou {self.acertos} de {len(self.letras_vogais)}")
            self.letra_atual = 0
            self.acertos = 0
    
    def voltar(self):
        self.timer.stop()
        self.cap.release()
        self.stack.setCurrentIndex(1)  # Volta para TelaNiveis