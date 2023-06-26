import sys
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tensorflow import lite
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QHBoxLayout, QDialog, QMenuBar, QAction, QMessageBox

# Importing the other files
from bin.aboutApp import aboutApp
from bin.help import Help

# Some variables that will be used through the program
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in ['J', 'Z']]   # List of all the alphabets that will be used
classes.append('_') # Represents whitespace
interpreter = tf.lite.Interpreter(model_path='models/model.tflite') # Loading the AI model
interpreter.allocate_tensors()

mpHands = mp.solutions.hands    # Initializing the hand tracking module
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Sign Language Translator")

        font = QFont()
        font.setPointSize(8)
        QApplication.setFont(font)

        self.icon = QIcon('icons/main.png')
        self.setWindowIcon(self.icon)

        # Some variables that will be used through the program
        self.processed_frames = pd.DataFrame(columns=[str(i) for i in range(25)] + ['label']) # Saves the processed characters which can improve the model
        self.text_field_stream = ''
        self.repeat_character_timer = 0     # Used when a character is repeating
        self.confidence_character = 0       # Used to ensure a random character prediction is not included, such as when transitioning between characters

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.image_label)

        self.text_field = QLineEdit()
        self.text_field.setPlaceholderText('Text will be display here')
        layout.addWidget(self.text_field)

        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.clearTextField)
        layout.addWidget(self.clear_button)

        self.label1 = QLabel('Was this response correct?')
        self.response_button_yes = QPushButton('Yes')
        self.response_button_yes.clicked.connect(self.saveProcessedFields)
        self.response_button_no = QPushButton('No')
        self.response_button_no.clicked.connect(self.discardProcessedFields)

        self.response_layout = QHBoxLayout()
        self.response_layout.addWidget(self.label1)
        self.response_layout.addWidget(self.response_button_yes)
        self.response_layout.addWidget(self.response_button_no)

        self.menubar = QMenuBar()
        self.model_list = self.menubar.addMenu('Model')
        self.update_model_option = QAction('Update Model', self)
        self.update_model_option.triggered.connect(self.update_model)
        self.model_list.addAction(self.update_model_option)

        self.train_model_option = QAction('Train this model', self)
        self.train_model_option.triggered.connect(self.train_model_clicked)
        self.model_list.addAction(self.train_model_option)

        self.help_list = self.menubar.addMenu('Help')
        self.about_option = QAction('About this App', self)
        self.about_option.triggered.connect(self.about)
        self.help_list.addAction(self.about_option)

        self.how_to_use_option = QAction('How to use this application?', self)
        self.how_to_use_option.triggered.connect(self.how_to_use)
        self.help_list.addAction(self.how_to_use_option)

        layout.setMenuBar(self.menubar)
        layout.addLayout(self.response_layout)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(100)

    def closeEvent(self, event):
        self.timer.stop()
        super(MainWindow, self).closeEvent(event)

    def clearTextField(self):
        self.text_field.setText('')
        self.text_field_stream = ''

    # Saving the predictions for retraining the model
    def saveProcessedFields(self):
        try:
            file = pd.read_csv('data/update_model.csv')
            self.processed_frames = self.processed_frames[file.columns]
            file = pd.concat([file, self.processed_frames], axis=0)
            file.to_csv('data/update_model.csv', index=False)
        except FileNotFoundError:
            columns = [str(i) for i in range(25)]+['label']
            file = pd.DataFrame(columns=columns)
            file = pd.concat([file, self.processed_frames], axis=0, ignore_index=True)
            file.to_csv('data/update_model.csv', index=False)
        except Exception as e:
            print(e)
        temp = pd.DataFrame(columns=[str(i) for i in range(25)] + ['label'])
        self.processed_frames = temp.copy()

    # Updating the AI model
    def update_model(self):

        train = pd.read_csv('data/update_model.csv')

        if train['label'].nunique() > 0:
            X_train = train.copy()
            train_y = X_train.pop('label')
            # Displaying a basic alert
            alert = QMessageBox()
            alert.setIcon(QMessageBox.Information)
            alert.setText("The system is retraining the model. This is likely to cause lag.")
            alert.setWindowTitle("Alert")
            alert.setStandardButtons(QMessageBox.Ok)
            alert.exec_()

            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder()
            train_y = oe.fit_transform(train_y.to_numpy().reshape(-1, 1))
            model = tf.keras.models.load_model('models/model.h5')
            model.fit(X_train, train_y, epochs=10)

            # Save the new model
            tf.keras.models.save_model(model, 'models/model.h5')

            # Save the TensorFlow Lite model
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open('models/model.tflite', 'wb') as f:
                f.write(tflite_model)

            # Reloading the model
            interpreter = tf.lite.Interpreter(model_path='models/model.tflite')
            interpreter.allocate_tensors()

            # Deleting update_model.csv
            train.drop(train.index, inplace=True)
            train.to_csv('data/update_model.csv', index=False)

            del train, X_train, train_y, model
        else:
            alert = QMessageBox()
            alert.setIcon(QMessageBox.Information)
            alert.setText("New data was not found to retrain the model..")
            alert.setWindowTitle("Alert")
            alert.setStandardButtons(QMessageBox.Ok)
            alert.exec_()

            del train

    # Training the model from scratch
    def train_model_clicked(self):
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setText("Please open the 'Train Model' file to train the model.")
        alert.setWindowTitle("Alert")
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()

    # Removing predictions if they are incorrent
    def discardProcessedFields(self):
        temp = pd.DataFrame(columns=[str(i) for i in range(25)] + ['label'])
        self.processed_frames = temp.copy()

    # Normalizing the co-ordinates of hand points
    def normalize(self, entry):
        ref = (entry.loc[0, '0_x'], entry.loc[0, '0_y'])

        hand_points = []
        for i in range(0, 21):
            hand_points.append((entry.loc[0, str(i) + '_x'], entry.loc[0, str(i) + '_y']))

        distances = []
        for point in hand_points:
            distance = math.sqrt((point[0] - ref[0]) ** 2 + (point[1] - ref[1]) ** 2)
            distances.append(distance)

        norm_distances = []
        ref2 = hand_points[12]
        palm_distance = math.sqrt((ref[0] - ref2[0]) ** 2 + (ref[1] - ref2[1]) ** 2)
        finger_distance = math.sqrt(
            (hand_points[8][0] - hand_points[12][0]) ** 2 + (hand_points[8][1] - hand_points[12][1]) ** 2)
        k1 = math.sqrt((hand_points[5][0] - hand_points[9][0]) ** 2 + (hand_points[5][1] - hand_points[9][1]) ** 2)
        k2 = math.sqrt((hand_points[13][0] - hand_points[17][0]) ** 2 + (hand_points[13][1] - hand_points[17][1]) ** 2)
        k3 = math.sqrt((hand_points[9][0] - hand_points[13][0]) ** 2 + (hand_points[9][1] - hand_points[13][1]) ** 2)
        for distance in distances:
            norm_distance = distance / palm_distance
            norm_distances.append(norm_distance)

        norm_distances.append(finger_distance / palm_distance)
        norm_distances.append(k1 / palm_distance)
        norm_distances.append(k2 / palm_distance)
        norm_distances.append(k3 / palm_distance)

        entry_final = pd.DataFrame()

        for i in range(0, 25):
            entry_final.loc[0, str(i)] = norm_distances[i]

        return entry_final

    # Making predictions
    def process_frame(self):
        success, img = self.cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            test = pd.DataFrame()
            for handVisible in results.multi_hand_landmarks:
                for id, lm in enumerate(handVisible.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    test.loc[0, str(id) + '_x'] = cx
                    test.loc[0, str(id) + '_y'] = cy

                mpDraw.draw_landmarks(img, handVisible, mpHands.HAND_CONNECTIONS)
            # Normalizing:
            test_final = self.normalize(test)
            input_data = np.array(test_final)
            input_data = input_data.astype(np.float32)
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            y_preds = tf.nn.softmax(output_data)
            if y_preds[0][np.argmax(y_preds)] > 0.0:    # Change this condition
                pred = classes[np.argmax(y_preds)]
                stream = pred

                if self.text_field_stream and self.text_field_stream[-1] == stream:
                    self.repeat_character_timer += 1
                elif self.repeat_character_timer > 10:
                    test_final['label'] = pred
                    self.processed_frames = pd.concat([self.processed_frames, test_final], axis=0, ignore_index=True)
                    if pred == '_':
                        self.text_field_stream += ' '
                    else:
                        self.text_field_stream += stream
                    self.text_field.setText(self.text_field_stream)
                    self.repeat_character_timer = 0
                else:
                    self.confidence_character += 1
                    if self.confidence_character > 10:
                        test_final['label'] = pred
                        self.processed_frames = pd.concat([self.processed_frames, test_final], axis=0, ignore_index=True)
                        self.confidence_character = 0
                        if pred == '_':
                            self.text_field_stream += ' '
                        else:
                            self.text_field_stream += stream
                        self.text_field.setText(self.text_field_stream)
                        self.repeat_character_timer = 0

        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def about(self):
        self.temp_window = aboutApp()
        self.temp_window.show()

    def how_to_use(self):
        self.temp_window = Help()
        self.temp_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
