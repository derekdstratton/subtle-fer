import os

import pandas as pd

#https://pythonprogramminglanguage.com/pyqt5-video-widget/
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
import sys

class EmotionButton(QPushButton):
    def __init__(self, name, vidWindow, styleSheet=None):
        super().__init__(name)
        self.name = name
        if styleSheet is not None:
            self.setStyleSheet(styleSheet)
        self.clicked.connect(self.emotion_button_click)
        self.vidWindow = vidWindow

    def emotion_button_click(self):
        print('button hit')
        # todo: this should save for the previous segment counted.
        self.vidWindow.segments['label'][self.vidWindow.segment_counter] = self.name

        startframe = self.vidWindow.segments['start_frame'][self.vidWindow.segment_counter]
        endframe = self.vidWindow.segments['end_frame'][self.vidWindow.segment_counter]
        # arg is milliseconds (assuming video is run at 30 fps)
        frames_to_millisecs = startframe // 30 * 1000
        self.vidWindow.mediaPlayer.setPosition(frames_to_millisecs)
        self.vidWindow.mediaPlayer.play()
        while self.vidWindow.mediaPlayer.position() * 30 // 1000 < endframe:
            print(self.vidWindow.mediaPlayer.position() * 30 // 1000)
        self.vidWindow.mediaPlayer.pause()

        self.vidWindow.segment_counter = self.vidWindow.segment_counter + 1
        if self.vidWindow.segment_counter >= len(self.vidWindow.segments):
            self.vidWindow.segment_counter = 0

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("PyQt Video Player Widget Example - pythonprogramminglanguage.com")

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.button = QPushButton('button', self)
        self.button.clicked.connect(self.on_click)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)

        # Create exit action
        # todo: link this to x button
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.button)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        hbox = QHBoxLayout()
        vbox1 = QVBoxLayout()
        vbox1.addWidget(EmotionButton("Amusement", self, styleSheet="background-color: yellow"))
        vbox1.addWidget(EmotionButton("Excitement", self, styleSheet="background-color: yellow"))
        vbox1.addWidget(EmotionButton("Surprise", self, styleSheet="background-color: yellow"))
        vbox1.addWidget(EmotionButton("Pride", self, styleSheet="background-color: cyan"))
        vbox1.addWidget(EmotionButton("Love", self, styleSheet="background-color: hotpink"))
        vbox1.addWidget(EmotionButton("Sexual Desire", self, styleSheet="background-color: hotpink"))

        vbox2 = QVBoxLayout()
        vbox2.addWidget(EmotionButton("Contentment", self, styleSheet="background-color: green"))
        vbox2.addWidget(EmotionButton("Relief", self, styleSheet="background-color: green"))
        vbox2.addWidget(EmotionButton("Happiness", self, styleSheet="background-color: green"))
        vbox2.addWidget(EmotionButton("Sadness", self, styleSheet="background-color: blue"))
        vbox2.addWidget(EmotionButton("Disappointment", self, styleSheet="background-color: blue"))
        vbox2.addWidget(EmotionButton("Guilt", self, styleSheet="background-color: blue"))

        vbox3 = QVBoxLayout()
        vbox3.addWidget(EmotionButton("Anger", self, styleSheet="background-color: red"))
        vbox3.addWidget(EmotionButton("Contempt", self, styleSheet="background-color: red"))
        vbox3.addWidget(EmotionButton("Pain", self, styleSheet="background-color: red"))
        vbox3.addWidget(EmotionButton("Jealousy", self, styleSheet="background-color: purple"))
        vbox3.addWidget(EmotionButton("Disgust",self,  styleSheet="background-color: purple"))

        vbox4 = QVBoxLayout()
        vbox4.addWidget(EmotionButton("Awkwardness", self, styleSheet="background-color: orange"))
        vbox4.addWidget(EmotionButton("Confusion", self, styleSheet="background-color: orange"))
        vbox4.addWidget(EmotionButton("Embarassment", self, styleSheet="background-color: orange"))
        vbox4.addWidget(EmotionButton("Fear", self, styleSheet="background-color: orange"))
        vbox4.addWidget(EmotionButton("Nervousness",self,  styleSheet="background-color: orange"))


        vbox5 = QVBoxLayout()
        vbox5.addWidget(EmotionButton("Neutral", self, styleSheet="background-color: gray"))
        vbox5.addWidget(EmotionButton("Boredom",self,  styleSheet="background-color: gray"))
        vbox5.addWidget(EmotionButton("Tired",self,  styleSheet="background-color: gray"))
        vbox5.addSpacing(20)
        vbox5.addWidget(EmotionButton("BAD CLIP",self ))

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        hbox.addLayout(vbox4)
        hbox.addLayout(vbox5)

        layout.addLayout(hbox)

        # layout.addWidget(EmotionButton("LOL"))

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

        video_path = "videos/simple_test.mp4"
        segments_path = "segment_labels/simple_test.csv"
        self.segments = pd.read_csv(segments_path)

        # open the media and segments
        # needs an absolute path
        self.mediaPlayer.setMedia(
            QMediaContent(QUrl.fromLocalFile(os.path.abspath(video_path))))
        self.playButton.setEnabled(True)

        self.segment_counter = 0

        # todo: onstart it should play the first clip, so you can label it with a button press.


    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())
        print(fileName)
        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
        self.segments = pd.read_csv("segment_labels/" + os.path.basename(fileName).split('.')[0] + ".csv")

    def exitCall(self):
        print('saving on exit')
        # todo: make this path dynamic for whatevedr file
        self.segments.to_csv("segment_labels/simple_test.csv")
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    def on_click(self):
        print('button hit')
        startframe = self.segments['start_frame'][self.segment_counter]
        endframe = self.segments['end_frame'][self.segment_counter]
        # arg is milliseconds (assuming video is run at 30 fps)
        frames_to_millisecs = startframe // 30 * 1000
        self.mediaPlayer.setPosition(frames_to_millisecs)
        self.mediaPlayer.play()
        while self.mediaPlayer.position() * 30 // 1000 < endframe:
            print(self.mediaPlayer.position() * 30 // 1000)
        self.mediaPlayer.pause()

        self.segment_counter = self.segment_counter + 1
        if self.segment_counter >= len(self.segments):
            self.segment_counter = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())