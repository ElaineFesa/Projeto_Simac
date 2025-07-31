from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

class TelaNiveis(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("🧩 Escolha um Nível"))
        
        btn_vogais = QPushButton("Nível 1 - Vogais")
        btn_vogais.clicked.connect(self.ir_para_intro_vogais)
        layout.addWidget(btn_vogais)

        self.setLayout(layout)

    def ir_para_intro_vogais(self):
        self.stack.setCurrentIndex(2)  # vai para a introdução do nível
