from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class TelaVogais(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()

        titulo = QLabel("Lição: Vogais em Libras")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 28px; color: #5E17EB;")

        botao_voltar = QPushButton("Voltar")
        botao_voltar.setStyleSheet("font-size: 20px; padding: 15px;")
        botao_voltar.clicked.connect(self.voltar_para_niveis)

        layout.addWidget(titulo)
        layout.addWidget(botao_voltar)
        self.setLayout(layout)

    def voltar_para_niveis(self):
        self.stack.setCurrentIndex(1)  # Volta para TelaNiveis
