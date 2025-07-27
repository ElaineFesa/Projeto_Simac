from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class TelaNiveis(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: #512D6D;")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        titulo = QLabel("Escolha um Nível")
        titulo.setFont(QFont("Arial", 28, QFont.Bold))
        titulo.setStyleSheet("color: white;")
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)
        layout.addSpacing(20)

        grid = QGridLayout()
        botoes = [
            ("Vogais", 0, 0),
            ("Consoantes (Parte 1)", 0, 1),
            ("Consoantes (Parte 2)", 1, 0),
            ("Números", 1, 1),
            ("Adjetivos", 2, 0),
            ("Cumprimentos", 2, 1),
        ]

        for texto, linha, coluna in botoes:
            btn = QPushButton(texto)
            btn.setFixedSize(300, 60)
            btn.setStyleSheet("background-color: #BB86FC; color: white; font-size: 18px; border-radius: 15px;")
            grid.addWidget(btn, linha, coluna)

        layout.addLayout(grid)
        layout.addSpacing(30)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.setFixedSize(150, 40)
        btn_voltar.setStyleSheet("background-color: #9A67EA; color: white; font-size: 16px; border-radius: 10px;")
        btn_voltar.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        layout.addWidget(btn_voltar, alignment=Qt.AlignCenter)
        self.setLayout(layout)
