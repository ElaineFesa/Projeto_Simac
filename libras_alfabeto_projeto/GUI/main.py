from PyQt5.QtWidgets import QApplication, QStackedWidget
from telas.TelaMenu import TelaMenu
from telas.TelaNiveis import TelaNiveis
from telas.TelaIntroNivel import TelaIntroNivel
from telas.TelaTesteNivel import TelaTesteNivel
import sys

app = QApplication(sys.argv)

stack = QStackedWidget()

tela_menu = TelaMenu(stack)
tela_niveis = TelaNiveis(stack)
tela_intro_nivel = TelaIntroNivel(stack)
tela_teste_nivel = TelaTesteNivel(stack)

stack.addWidget(tela_menu)          # índice 0
stack.addWidget(tela_niveis)        # índice 1
stack.addWidget(tela_intro_nivel)   # índice 2
stack.addWidget(tela_teste_nivel)   # índice 3

stack.setCurrentIndex(0)
stack.setFixedSize(1024, 600)
stack.show()

sys.exit(app.exec_())
