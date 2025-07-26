# gui/telas/tela_menu.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class TelaMenu(QWidget):
    def __init__(self, trocar_tela_callback):
        super().__init__()
        self.trocar_tela_callback = trocar_tela_callback
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("background-color: #3D2C8D; color: white;")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        titulo = QLabel("Bem-vindo ao Aprenda Libras!")
        titulo.setFont(QFont("Arial", 28, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)

        btn_iniciar = QPushButton("Iniciar Aprendizado")
        btn_iniciar.setFixedHeight(50)
        btn_iniciar.setStyleSheet("background-color: #8F43EE; font-size: 20px;")

        btn_iniciar.clicked.connect(lambda: self.trocar_tela_callback("niveis"))

        layout.addWidget(titulo)
        layout.addSpacing(40)
        layout.addWidget(btn_iniciar)

        self.setLayout(layout)
