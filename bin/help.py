from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, \
    QHBoxLayout, QMessageBox, QFrame, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys


class Help(QMainWindow):

    def __init__(self):
        super(Help, self).__init__()
        self.resize(800,400)
        self.setWindowTitle('How to use this application?')

        self.icon = QIcon('../icons/info.png')
        self.setWindowIcon(self.icon)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

        self.nav = QVBoxLayout()

        self.option1 = QPushButton('How to use this app to translate?')
        self.option1.clicked.connect(self.option1_clicked)
        self.option1.setEnabled(False)
        self.option2 = QPushButton('How to make the model more accurate?')
        self.option2.clicked.connect(self.option2_clicked)
        self.option3 = QPushButton('How to retrain the model from scratch?')
        self.option3.clicked.connect(self.option3_clicked)

        self.nav.addWidget(self.option1)
        self.nav.addWidget(self.option2)
        self.nav.addWidget(self.option3)
        self.nav.addStretch(1)

        self.main_display = QWidget()
        self.main_layout = QVBoxLayout(self.main_display)
        self.main_layout.setContentsMargins(50, 10, 50, 10)

        self.display = QLabel()
        self.display.setWordWrap(True)
        self.display.adjustSize()
        self.display.setText(
            'The app will access your webcam and translate the sign language to text. The app will display an interface over a hand if it detects'
            ' it. It will then display the translated text in the text box.')
        self.main_layout.addWidget(self.display)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setWidget(self.main_display)

        seperator = QFrame()
        seperator.setFrameShape(QFrame.VLine)
        seperator.setFrameShadow(QFrame.Sunken)
        seperator.setLineWidth(3)
        seperator.setStyleSheet("color: rgb(0, 0, 0);")

        layout.addLayout(self.nav, 0, 0, 1, 1)
        layout.addWidget(seperator, 0, 1, 1, 1)
        layout.addWidget(scroll_area, 0, 2, 1, 4)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

    def option1_clicked(self):
        self.display.setText(
            'The app will access your webcam and translate the sign language to text. The app will display an interface over a hand if it detects'
            ' it. It will then display the translated text in the text box.')

        self.option1.setEnabled(False)
        self.option2.setEnabled(True)
        self.option3.setEnabled(True)

    def option2_clicked(self):
        self.display.setText(
            'When a response is accurate, respond \'yes\' to the prompt below the text field. Under the model option, click on retrain model to refresh the model'
            ' with the new data. The model will be retrained with the new data and will be more accurate. ')

        self.option1.setEnabled(True)
        self.option2.setEnabled(False)
        self.option3.setEnabled(True)

    def option3_clicked(self):
        self.display.setText(
            'Under the model option, click on update model, which will then train the model from scratch. This will take a while.')

        self.option1.setEnabled(True)
        self.option2.setEnabled(True)
        self.option3.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Help()
    window.show()
    sys.exit(app.exec_())
