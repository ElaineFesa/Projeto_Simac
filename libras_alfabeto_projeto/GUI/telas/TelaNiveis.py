from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class TelaNiveis(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.init_ui()
    
    def init_ui(self):
        self.setStyleSheet("background-color: #34495e; color: white;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        titulo = QLabel("Escolha um Nível")
        titulo.setStyleSheet("font-size: 24px;")
        
        btn_vogais = QPushButton("Vogais (Nível 1)")
        btn_vogais.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 18px;
                padding: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_vogais.clicked.connect(self.iniciar_vogais)
        
        btn_voltar = QPushButton("Voltar ao Menu")
        btn_voltar.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        
        layout.addWidget(titulo)
        layout.addWidget(btn_vogais)
        layout.addWidget(btn_voltar)
        
        self.setLayout(layout)
    
    def iniciar_vogais(self):
        self.stack.setCurrentIndex(2)  # Vai para TelaVogais