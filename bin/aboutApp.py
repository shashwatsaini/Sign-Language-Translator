from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys

# Class for the about app window
class aboutApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 400)
        self.setWindowTitle('About this App')

        self.icon = QIcon('../icons/info.png')
        self.setWindowIcon(self.icon)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.display1 = QLabel()
        self.display1.setWordWrap(True)
        self.display1.setContentsMargins(50, 10, 50, 10)
        self.display1.adjustSize()
        self.display1.setAlignment(Qt.AlignCenter)
        self.display1.setText('This app is a prototype for a sign language translator. It is made by Shashwat Saini, a student of '
                             'Dayananda Sagar University, Bangalore.')

        self.display2 = QLabel()
        self.display2.setContentsMargins(50, 10, 50, 10)
        self.display2.setAlignment(Qt.AlignCenter)
        self.display2.setOpenExternalLinks(True)
        self.display2.setText('Sign Language Translator © 2023 by Shashwat Saini is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1">CC BY-NC-SA 4.0</a>')

        self.display3 = QLabel()
        self.display3.setContentsMargins(50, 10, 50, 10)
        self.display3.setAlignment(Qt.AlignCenter)
        self.display3.setOpenExternalLinks(True)
        self.display3.setText('Follow me on Github: <a href="https://github.com/shashwatsaini">Shashwat Saini</a>')

        layout.addWidget(self.display1)
        layout.addWidget(self.display2)
        layout.addWidget(self.display3)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = aboutApp()
    window.show()
    sys.exit(app.exec_())