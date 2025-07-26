# gui/telas/tela_coleta.py

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

class TelaColeta(QWidget):
    voltar_menu = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coleta de Gestos")
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        titulo = QLabel("📷 Coleta de Dados - Gestos em Libras")
        titulo.setFont(QFont("Arial", 22))
        titulo.setAlignment(Qt.AlignCenter)

        instrucoes = QLabel("Escolha o gesto que deseja coletar:")
        instrucoes.setFont(QFont("Arial", 14))
        instrucoes.setAlignment(Qt.AlignCenter)

        self.combo_gesto = QComboBox()
        self.combo_gesto.setFont(QFont("Arial", 14))
        self.combo_gesto.addItems([
            "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "J",
            "1", "2", "3", "4", "5",
            "Oi", "Tchau", "Obrigado"
        ])

        self.label_status = QLabel("Status: Aguardando início da coleta")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("color: gray; font-size: 14px;")

        botoes = QHBoxLayout()
        self.btn_iniciar = QPushButton("▶️ Iniciar Coleta")
        self.btn_parar = QPushButton("⏹️ Parar Coleta")
        self.btn_voltar = QPushButton("🔙 Voltar")

        for btn in [self.btn_iniciar, self.btn_parar, self.btn_voltar]:
            btn.setFont(QFont("Arial", 14))
            btn.setMinimumHeight(40)

        self.btn_parar.setEnabled(False)

        botoes.addWidget(self.btn_iniciar)
        botoes.addWidget(self.btn_parar)
        botoes.addWidget(self.btn_voltar)

        # Aqui no futuro você pode integrar um QLabel para a câmera com OpenCV
        self.label_camera = QLabel("🖼️ Visualização da câmera aqui")
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setStyleSheet("border: 2px dashed #aaa; padding: 20px; font-size: 16px;")

        layout.addWidget(titulo)
        layout.addWidget(instrucoes)
        layout.addWidget(self.combo_gesto)
        layout.addWidget(self.label_camera)
        layout.addWidget(self.label_status)
        layout.addLayout(botoes)

        self.setLayout(layout)

        # Conexões de botões
        self.btn_voltar.clicked.connect(self.voltar_menu.emit)
        # A lógica da coleta será conectada depois com o módulo de vídeo
