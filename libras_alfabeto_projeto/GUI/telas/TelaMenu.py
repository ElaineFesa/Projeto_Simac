from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class TelaMenu(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.init_ui()
    
    def init_ui(self):
        self.setStyleSheet("background-color: #3F2A56; color: white;")
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        titulo = QLabel("Jogo de Libras")
        titulo.setFont(QFont("Arial", 32, QFont.Bold))
        
        btn_iniciar = QPushButton("Iniciar Jogo")
        btn_iniciar.setStyleSheet("""
            QPushButton {
                background-color: #A259FF;
                color: white;
                font-size: 20px;
                padding: 15px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #B388EB;
            }
        """)
        btn_iniciar.clicked.connect(self.ir_para_niveis)
        
        layout.addWidget(titulo)
        layout.addWidget(btn_iniciar)
        
        self.setLayout(layout)
    
    def ir_para_niveis(self):
        self.stack.setCurrentIndex(1)  # Vai para TelaNiveis