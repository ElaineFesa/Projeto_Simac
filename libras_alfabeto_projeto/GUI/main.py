import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from telas.TelaMenu import TelaMenu
from telas.TelaNiveis import TelaNiveis
from telas.TelaVogais import TelaVogais

app = QApplication(sys.argv)
stack = QStackedWidget()

# Telas
tela_menu = TelaMenu(stack)
tela_niveis = TelaNiveis(stack)
tela_vogais = TelaVogais(stack)  # Tela específica para vogais

# Adiciona telas
stack.addWidget(tela_menu)
stack.addWidget(tela_niveis)
stack.addWidget(tela_vogais)

stack.setCurrentWidget(tela_menu)
stack.setWindowTitle("Jogo de Libras - Aprenda Vogais")
stack.showMaximized()

sys.exit(app.exec_())