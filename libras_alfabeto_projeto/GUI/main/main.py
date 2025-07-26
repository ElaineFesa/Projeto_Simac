import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from ..telas.TelaPrincipal import TelaPrincipal

def aplicar_estilo_roxo(app):
    """Define uma paleta de cores com tons de roxo."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#2e003e"))
    palette.setColor(QPalette.WindowText, QColor("#ffffff"))
    palette.setColor(QPalette.Base, QColor("#3c1361"))
    palette.setColor(QPalette.AlternateBase, QColor("#5e239d"))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.Text, QColor("#ffffff"))
    palette.setColor(QPalette.Button, QColor("#5e239d"))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor("#ff80ff"))

    app.setStyle("Fusion")
    app.setPalette(palette)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    aplicar_estilo_roxo(app)  # Aplicar o tema roxo

    janela = TelasPrincipal()
    janela.showFullScreen()  # Abrir em tela cheia direto

    sys.exit(app.exec_())
