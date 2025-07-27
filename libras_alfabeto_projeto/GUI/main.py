import sys
from PyQt5.QtWidgets import QApplication, QStackedWidget
from telas.TelaMenu import TelaMenu
from telas.TelaNiveis import TelaNiveis

app = QApplication(sys.argv)
stack = QStackedWidget()

# Instanciando as telas
tela_menu = TelaMenu(stack)
tela_niveis = TelaNiveis(stack)

# Adicionando ao stack
stack.addWidget(tela_menu)
stack.addWidget(tela_niveis)

stack.setFixedSize(1280, 720)
stack.setWindowTitle("Plataforma Educativa de Libras")
stack.setCurrentWidget(tela_menu)
stack.show()

sys.exit(app.exec_())