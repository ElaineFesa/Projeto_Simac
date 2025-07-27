from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt

class TelaMenu(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.setStyleSheet("background-color: #3F2A56;")

        titulo = QLabel("Plataforma de Aprendizado em Libras")
        titulo.setFont(QFont("Arial", 32, QFont.Bold))
        titulo.setStyleSheet("color: white;")
        titulo.setAlignment(Qt.AlignCenter)

        btn_iniciar = QPushButton("Iniciar")
        btn_iniciar.setFixedHeight(60)
        btn_iniciar.setStyleSheet("background-color: #A259FF; color: white; font-size: 20px; border-radius: 20px;")
        btn_iniciar.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        layout.addWidget(titulo)
        layout.addSpacing(60)
        layout.addWidget(btn_iniciar)

        self.setLayout(layout)