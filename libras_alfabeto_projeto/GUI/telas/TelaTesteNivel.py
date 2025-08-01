from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class TelaTesteNivel(QWidget):
    def __init__(self, stack):
        super().__init__()  # Chama o construtor da classe base QWidget
        self.stack = stack  # Armazena a referência ao QStackedWidget
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Título
        titulo = QLabel("Teste de Nível")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("font-size: 24px; color: #5E17EB;")
        
        # Botão Voltar
        self.btn_voltar = QPushButton("Voltar")
        self.btn_voltar.setStyleSheet("font-size: 20px; padding: 15px;")
        self.btn_voltar.clicked.connect(self.voltar_callback)
        
        layout.addWidget(titulo)
        layout.addWidget(self.btn_voltar)
        self.setLayout(layout)
    
    def voltar_callback(self):
        self.stack.setCurrentIndex(1)  # Volta para TelaNiveis