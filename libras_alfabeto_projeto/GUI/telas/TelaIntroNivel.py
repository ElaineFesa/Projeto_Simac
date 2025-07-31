from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

class TelaIntroNivel(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

        layout = QVBoxLayout()

        layout.addWidget(QLabel("📖 Neste nível você aprenderá as vogais em Libras:\nA, E, I, O, U"))

        btn_avancar = QPushButton("Avançar para Teste")
        btn_avancar.clicked.connect(self.ir_para_teste)
        layout.addWidget(btn_avancar)

        self.setLayout(layout)

    def ir_para_teste(self):
        self.stack.setCurrentIndex(3)
