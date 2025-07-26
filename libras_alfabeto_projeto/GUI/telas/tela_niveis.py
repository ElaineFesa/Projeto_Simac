# gui/telas/tela_niveis.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class TelaNiveis(QWidget):
    def __init__(self, trocar_tela_callback):
        super().__init__()
        self.trocar_tela_callback = trocar_tela_callback
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: #5C3D99; color: white;")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        titulo = QLabel("Escolha um Nível")
        titulo.setFont(QFont("Arial", 24, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)

        grade = QGridLayout()
        grade.setSpacing(20)

        niveis = ["Vogais", "Consoantes", "Números", "Adjetivos", "Cumprimentos"]
        for i, nivel in enumerate(niveis):
            botao = QPushButton(nivel)
            botao.setFixedSize(200, 60)
            botao.setStyleSheet("background-color: #A267E7; font-size: 18px;")
            # Aqui você liga o botão ao nível correspondente
            botao.clicked.connect(lambda _, n=nivel: self.trocar_tela_callback("jogo", n))
            grade.addWidget(botao, i // 2, i % 2)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.setFixedHeight(40)
        btn_voltar.setStyleSheet("background-color: #D5B4FF; font-size: 16px;")
        btn_voltar.clicked.connect(lambda: self.trocar_tela_callback("menu"))

        layout.addWidget(titulo)
        layout.addSpacing(30)
        layout.addLayout(grade)
        layout.addSpacing(20)
        layout.addWidget(btn_voltar)

        self.setLayout(layout)
