from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

class TelaMenu(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        layout = QVBoxLayout()

        layout.addWidget(QLabel("📚 Bem-vindo ao Sistema de Aprendizado de Libras"))
        
        btn_niveis = QPushButton("Iniciar Níveis")
        btn_niveis.clicked.connect(self.ir_para_niveis)
        layout.addWidget(btn_niveis)

        self.setLayout(layout)

    def ir_para_niveis(self):
        self.stack.setCurrentIndex(1)
