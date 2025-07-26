import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QStackedWidget, QWidget,
    QVBoxLayout, QLabel, QHBoxLayout, QListWidget
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt


class TelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Libras Jogo Educativo")
        self.showFullScreen()  # TELA CHEIA

        # Paleta roxa
        self.setPalette(self.criar_paleta_roxa())

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.tela_menu = TelaMenu(self)
        self.tela_niveis = TelaNiveis(self)
        self.tela_nivel = TelaNivel(self)

        self.stacked_widget.addWidget(self.tela_menu)
        self.stacked_widget.addWidget(self.tela_niveis)
        self.stacked_widget.addWidget(self.tela_nivel)

        self.mudar_tela("menu")

    def criar_paleta_roxa(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#2e003e"))
        palette.setColor(QPalette.WindowText, QColor("#ffffff"))
        palette.setColor(QPalette.Base, QColor("#3c1361"))
        palette.setColor(QPalette.Button, QColor("#5e239d"))
        palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
        palette.setColor(QPalette.Text, QColor("#ffffff"))
        return palette

    def mudar_tela(self, tela):
        if tela == "menu":
            self.stacked_widget.setCurrentWidget(self.tela_menu)
        elif tela == "niveis":
            self.stacked_widget.setCurrentWidget(self.tela_niveis)
        elif tela == "nivel":
            self.stacked_widget.setCurrentWidget(self.tela_nivel)


class TelaMenu(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        titulo = QLabel("Bem-vindo ao Libras Educacional")
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setFont(QFont("Arial", 36, QFont.Bold))
        layout.addWidget(titulo)

        btn_jogar = QPushButton("Iniciar Jogo")
        btn_jogar.setFixedHeight(60)
        btn_jogar.setFont(QFont("Arial", 16))
        btn_jogar.clicked.connect(lambda: controller.mudar_tela("niveis"))
        layout.addWidget(btn_jogar)

        # SOMENTE VISÍVEL PARA DEV (trocar para True se quiser mostrar)
        mostrar_botao_coleta = False
        if mostrar_botao_coleta:
            btn_coletar = QPushButton("Coletar Dados")
            btn_coletar.setFont(QFont("Arial", 14))
            btn_coletar.clicked.connect(self.abrir_coleta)
            layout.addWidget(btn_coletar)

        self.setLayout(layout)

    def abrir_coleta(self):
        print("Abrindo coleta...")


class TelaNiveis(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        titulo = QLabel("Escolha um Nível")
        titulo.setFont(QFont("Arial", 30, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        self.lista_niveis = QListWidget()
        self.lista_niveis.setFont(QFont("Arial", 16))
        self.lista_niveis.addItems([
            "Vogais",
            "Consoantes - Parte 1",
            "Consoantes - Parte 2",
            "Números",
            "Adjetivos",
            "Cumprimentos"
        ])
        layout.addWidget(self.lista_niveis)

        botoes = QHBoxLayout()

        btn_iniciar = QPushButton("Iniciar Nível")
        btn_iniciar.setFixedHeight(40)
        btn_iniciar.clicked.connect(self.iniciar_nivel)
        botoes.addWidget(btn_iniciar)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.setFixedHeight(40)
        btn_voltar.clicked.connect(lambda: controller.mudar_tela("menu"))
        botoes.addWidget(btn_voltar)

        layout.addLayout(botoes)
        self.setLayout(layout)

    def iniciar_nivel(self):
        item = self.lista_niveis.currentItem()
        if item:
            self.controller.tela_nivel.set_titulo(item.text())
            self.controller.mudar_tela("nivel")


class TelaNivel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label_titulo = QLabel("Nível")
        self.label_titulo.setFont(QFont("Arial", 28, QFont.Bold))
        self.label_titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_titulo)

        self.label_instrucao = QLabel("Clique em 'Iniciar reconhecimento' para testar o gesto")
        self.label_instrucao.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_instrucao)

        botoes = QHBoxLayout()

        btn_reconhecer = QPushButton("Iniciar reconhecimento")
        btn_reconhecer.clicked.connect(self.reconhecer)
        botoes.addWidget(btn_reconhecer)

        btn_voltar = QPushButton("Voltar")
        btn_voltar.clicked.connect(lambda: controller.mudar_tela("niveis"))
        botoes.addWidget(btn_voltar)

        layout.addLayout(botoes)
        self.setLayout(layout)

    def set_titulo(self, texto):
        self.label_titulo.setText(f"Nível: {texto}")

    def reconhecer(self):
        print("Reconhecendo gesto...")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = TelaPrincipal()
    janela.show()
    sys.exit(app.exec_())
