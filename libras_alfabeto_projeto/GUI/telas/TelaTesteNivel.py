from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import time

class TelaTesteNivel(QWidget):
    def __init__(self, voltar_callback):
        super().__init__()
        self.voltar_callback = voltar_callback

        self.letras_nivel = ['A', 'E', 'I', 'O', 'U']
        self.letra_atual = 0
        self.erros = []

        self.setWindowTitle("Teste de Nível - Vogais")
        self.layout = QVBoxLayout()

        self.label = QLabel(f"Mostre a letra em Libras: {self.letras_nivel[self.letra_atual]}")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.btn_comecar = QPushButton("Começar")
        self.btn_comecar.clicked.connect(self.reconhecer_gesto)
        self.layout.addWidget(self.btn_comecar)

        self.btn_voltar = QPushButton("Voltar")
        self.btn_voltar.clicked.connect(self.voltar_callback)
        self.layout.addWidget(self.btn_voltar)

        self.setLayout(self.layout)

        # carregar modelo e encoder
        self.modelo = load_model("modelos/modelo_libras.h5")
        with open("modelos/labels.pkl", "rb") as f:
            self.encoder = pickle.load(f)

        # mediapipe
        self.hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

    def reconhecer_gesto(self):
        cap = cv2.VideoCapture(0)
        frames_coletados = []

        start_time = time.time()
        while time.time() - start_time < 3:  # 3 segundos para coletar dados
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    pontos = []
                    for lm in hand_landmarks.landmark:
                        pontos.extend([lm.x, lm.y, lm.z])
                    if len(pontos) == 63:  # 21 pontos * 3 coords
                        frames_coletados.append(pontos)
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            cv2.putText(frame, "Mostre o gesto da letra", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Reconhecimento", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(frames_coletados) < 10:
            QMessageBox.warning(self, "Erro", "Não foi possível capturar o gesto corretamente.")
            return

        sequencia = frames_coletados[:30]
        while len(sequencia) < 30:
            sequencia.append([0.0]*63)

        entrada = np.array([sequencia])
        pred = self.modelo.predict(entrada)
        classe = self.encoder.inverse_transform([np.argmax(pred)])

        letra_esperada = self.letras_nivel[self.letra_atual]
        if classe[0] == letra_esperada:
            QMessageBox.information(self, "Correto!", f"Você acertou a letra {letra_esperada}")
        else:
            QMessageBox.warning(self, "Incorreto", f"Você fez {classe[0]}, mas a letra correta era {letra_esperada}")
            self.erros.append(letra_esperada)

        self.letra_atual += 1
        if self.letra_atual >= len(self.letras_nivel):
            self.finalizar_teste()
        else:
            self.label.setText(f"Mostre a letra em Libras: {self.letras_nivel[self.letra_atual]}")

    def finalizar_teste(self):
        if self.erros:
            msg = "Você errou as letras: " + ", ".join(self.erros) + ". Deseja tentar novamente?"
            resposta = QMessageBox.question(self, "Finalizado", msg, QMessageBox.Yes | QMessageBox.No)
            if resposta == QMessageBox.Yes:
                self.letra_atual = 0
                self.erros = []
                self.label.setText(f"Mostre a letra em Libras: {self.letras_nivel[self.letra_atual]}")
            else:
                self.voltar_callback()
        else:
            QMessageBox.information(self, "Parabéns!", "Você acertou todas as letras!")
            self.voltar_callback()
