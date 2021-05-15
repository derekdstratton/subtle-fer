import os

import pandas as pd

#https://pythonprogramminglanguage.com/pyqt5-video-widget/
from PyQt5 import QtCore
from PyQt5.QtCore import QDir, Qt, QUrl, QTimer, QPoint, QRect, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QGridLayout)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QPainter, QPen, QPaintDevice, QBrush, QPolygon, QImage
import sys

# kind of a hacky way, just put them in a list of strings
emotions_string_list = []

import functools

# this is a clever solution to patch up my bad design
# https://stackoverflow.com/questions/55553660/how-to-emit-custom-events-to-the-event-loop-in-pyqt
@functools.lru_cache()
class GlobalObject(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self._events = {}

    def addEventListener(self, name, func):
        if name not in self._events:
            self._events[name] = [func]
        else:
            self._events[name].append(func)

    def dispatchEvent(self, name):
        functions = self._events.get(name, [])
        for func in functions:
            QtCore.QTimer.singleShot(0, func)

# class VideoWidgetOverride(QVideoWidget):
#     def __init__(self):
#         super().__init__()
#         self.points = QPolygon()
#         self.draw = False
#         self.setMouseTracking(True)
#
#     def mousePressEvent(self, e):
#         self.points << e.pos()
#         self.draw = True
#         self.update()
#         # self.repaint()
#
#     def mouseMoveEvent(self, e):
#         print(e.pos())
#         if self.draw:
#             self.points << e.pos()
#             self.update()
#             print(e.pos())
#
#     def mouseReleaseEvent(self, e):
#         self.draw = False
#
#     def paintEvent(self, ev):
#         # super().paintEvent(ev)
#         qp = QPainter(self)
#         qp.setRenderHint(QPainter.Antialiasing)
#         pen = QPen(Qt.red, 5)
#         brush = QBrush(Qt.red)
#         qp.setPen(pen)
#         qp.setBrush(brush)
#         for i in range(self.points.count()):
#             qp.drawEllipse(self.points.point(i), 5, 5)
#         # https://stackoverflow.com/questions/4372195/cant-override-videowidget-paintevent-in-qt-c
#         # self.update()

class DonePopup(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Congratulations! You've finished labeling. Please follow the submission instructions\nto send us your completed labels. This application is safe to close now."))
        self.setLayout(hbox)


# https://doc.qt.io/qt-5/qtwidgets-widgets-scribble-example.html
class Overlay(QWidget):
    def __init__(self, x, y, width, height):
        super().__init__()
        # self.setAutoFillBackground(False)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        # self.setStyleSheet("background-color: transparent")
        # self.setStyleSheet("QWidget{background: #000000}")
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setWindowOpacity(3)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.setMouseTracking(True)

        self.hasBeenDrawnOn = False
        # self.setGeometry(x, y, width, height)
        # self.setFixedSize(width, height)
        # self.im = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        # print(self.width())
        # print(self.height())
        # self.im.fill(Qt.transparent)
        # self.update()

    def mousePressEvent(self, e):
        self.hasBeenDrawnOn = True
        if e.button() == Qt.LeftButton:
            self.lastPoint = e.pos()
            self.scribbling = True

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and self.scribbling:
            self.drawLineTo(e.pos())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(e.pos())
            self.scribbling = False

    def drawLineTo(self, endPoint):
        painter = QPainter(self.im)
        myPenColor = Qt.red
        myPenWidth = 5
        painter.setPen(QPen(myPenColor, myPenWidth, Qt.SolidLine, Qt.RoundCap,Qt.RoundJoin))
        painter.setOpacity(0.3)
        painter.drawLine(self.lastPoint, endPoint)
        modified = True

        rad = int((myPenWidth / 2) + 2)
        # self.update()
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = endPoint

    def paintEvent(self, ev):
        qp = QPainter(self)
        drect = ev.rect()
        qp.drawImage(drect, self.im, drect)

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_S:
    #         self.im.save("LOL.jpg", "jpg")
    #     elif event.key() == Qt.Key_C:
    #         self.im.fill(Qt.transparent)
    #         self.update()

    def showEvent(self, event):
        self.setGeometry(self.mainX, self.mainY, self.mainWid, self.mainHei)
        self.setFixedSize(self.mainWid, self.mainHei)
        self.im = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        print(self.width())
        print(self.height())
        self.im.fill(Qt.transparent)
        self.update()

    # def paintEvent(self, ev):
    #     # super().paintEvent(ev)
    #     qp = QPainter(self)
    #     qp.setRenderHint(QPainter.Antialiasing)
    #     qp.setOpacity(0.2)
    #     # pen = QPen(Qt.red, 5)
    #     pen = QPen(Qt.NoPen)
    #     brush = QBrush(Qt.red)#, Qt.Dense5Pattern)
    #     qp.setPen(pen)
    #     qp.setBrush(brush)
    #     for i in range(self.points.count()):
    #         qp.drawEllipse(self.points.point(i), 5, 5)
    #         # qp.drawPoint(self.points.point(i))
    #     # https://stackoverflow.com/questions/4372195/cant-override-videowidget-paintevent-in-qt-c
    #     # self.update()

class EmotionButton(QPushButton):
    def __init__(self, name, vidWindow, styleSheet=None):
        super().__init__(name)
        self.name = name
        if styleSheet is not None:
            self.setStyleSheet(styleSheet)
        self.clicked.connect(self.emotion_button_click)
        self.vidWindow = vidWindow

        # self.vidWindow.submit_button.clicked.connect(self.pls)
        GlobalObject().addEventListener("success", self.pls)

        self.vidWindow.back.clicked.connect(self.pls)

        # toggle
        # https://www.geeksforgeeks.org/pyqt5-toggle-button/
        self.setCheckable(True)

    def emotion_button_click(self):
        print('button hit: ' + self.name)
        # self.vidWindow.segments['label'][self.vidWindow.segment_counter] = self.name
        # TODO: label the data here
        if self.isChecked():
            emotions_string_list.append(self.name)
            self.setStyleSheet("background-color: blue")
        else:
            emotions_string_list.remove(self.name)
            self.setStyleSheet("background-color: gray")

        # self.vidWindow.df['label'][self.vidWindow.segment_counter] = self.name
        #
        # if self.vidWindow.segment_counter >= len(self.vidWindow.files)-1:
        #     print("YOURE FINISHED HERE: AUTOSAVING.")
        #     self.vidWindow.segmentLabel.setStyleSheet("color:green")
        #     self.vidWindow.df.to_csv(self.vidWindow.labels_path)
        #     return
        # self.vidWindow.segment_counter = self.vidWindow.segment_counter + 1
        # self.vidWindow.segmentLabel.setText(str(self.vidWindow.segment_counter) + "/" + str(len(self.vidWindow.files)-1))
        # # self.vidWindow.playsegment(self.vidWindow.segment_counter)
        # self.vidWindow.simpleplay(self.vidWindow.segment_counter)

    def pls(self):
        emotions_string_list.clear()
        self.setChecked(False)
        self.setStyleSheet("background-color: gray")

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)

        self.drawingRef = None

        self.setWindowState(Qt.WindowActive)

        self.setWindowTitle("Emotion Segment Labeler")

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # self.video_widget = VideoWidgetOverride()
        self.video_widget = QVideoWidget()

        self.segment_counter = 0

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.replay = QPushButton('replay', self)
        self.replay.clicked.connect(lambda: self.simpleplay(self.segment_counter))

        self.back = QPushButton('back', self)
        self.back.clicked.connect(self.goback)

        self.clear = QPushButton('clear', self)
        self.clear.clicked.connect(self.clear_drawing)

        self.segmentLabel = QLabel("")

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # removing these for now, since I have autosave, and it opens the 1 dir
        # if you want to start over, just delete the labels.csv file.

        # Create new action
        # openAction = QAction(QIcon('open.png'), '&Open', self)
        # openAction.setShortcut('Ctrl+O')
        # openAction.setStatusTip('Open folder')
        # openAction.triggered.connect(self.openFile)

        # saving action
        # saveAction = QAction(QIcon('save.png'), '&Save', self)
        # saveAction.setShortcut('Ctrl+S')
        # saveAction.setStatusTip('Save labels')
        # saveAction.triggered.connect(self.saveFile)

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
        # fileMenu.addAction(openAction)
        # fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        # controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.segmentLabel)
        controlLayout.addWidget(self.back)
        controlLayout.addWidget(self.replay)
        controlLayout.addWidget(self.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)

        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        hbox = QHBoxLayout()
        # colored
        # vbox1 = QVBoxLayout()
        # vbox1.addWidget(EmotionButton("Amusement", self, styleSheet="background-color: yellow"))
        # vbox1.addWidget(EmotionButton("Excitement", self, styleSheet="background-color: yellow"))
        # vbox1.addWidget(EmotionButton("Surprise", self, styleSheet="background-color: yellow"))
        # vbox1.addWidget(EmotionButton("Pride", self, styleSheet="background-color: cyan"))
        # vbox1.addWidget(EmotionButton("Love", self, styleSheet="background-color: hotpink"))
        # vbox1.addWidget(EmotionButton("Sexual Desire", self, styleSheet="background-color: hotpink"))
        #
        # vbox2 = QVBoxLayout()
        # vbox2.addWidget(EmotionButton("Contentment", self, styleSheet="background-color: green"))
        # vbox2.addWidget(EmotionButton("Relief", self, styleSheet="background-color: green"))
        # vbox2.addWidget(EmotionButton("Happiness", self, styleSheet="background-color: green"))
        # vbox2.addWidget(EmotionButton("Sadness", self, styleSheet="background-color: blue"))
        # vbox2.addWidget(EmotionButton("Disappointment", self, styleSheet="background-color: blue"))
        # vbox2.addWidget(EmotionButton("Guilt", self, styleSheet="background-color: blue"))
        #
        # vbox3 = QVBoxLayout()
        # vbox3.addWidget(EmotionButton("Anger", self, styleSheet="background-color: red"))
        # vbox3.addWidget(EmotionButton("Contempt", self, styleSheet="background-color: red"))
        # vbox3.addWidget(EmotionButton("Pain", self, styleSheet="background-color: red"))
        # vbox3.addWidget(EmotionButton("Jealousy", self, styleSheet="background-color: purple"))
        # vbox3.addWidget(EmotionButton("Disgust",self,  styleSheet="background-color: purple"))
        #
        # vbox4 = QVBoxLayout()
        # vbox4.addWidget(EmotionButton("Awkwardness", self, styleSheet="background-color: orange"))
        # vbox4.addWidget(EmotionButton("Confusion", self, styleSheet="background-color: orange"))
        # vbox4.addWidget(EmotionButton("Embarassment", self, styleSheet="background-color: orange"))
        # vbox4.addWidget(EmotionButton("Fear", self, styleSheet="background-color: orange"))
        # vbox4.addWidget(EmotionButton("Nervousness",self,  styleSheet="background-color: orange"))
        #
        #
        # vbox5 = QVBoxLayout()
        # vbox5.addWidget(EmotionButton("Neutral", self, styleSheet="background-color: gray"))
        # vbox5.addWidget(EmotionButton("Boredom",self,  styleSheet="background-color: gray"))
        # vbox5.addWidget(EmotionButton("Tired",self,  styleSheet="background-color: gray"))
        # vbox5.addSpacing(20)
        # vbox5.addWidget(EmotionButton("BAD CLIP",self ))

        self.submit_button = QPushButton("submit")
        self.submit_button.clicked.connect(self.submit)

        # black and white
        vbox1 = QVBoxLayout()
        vbox1.addWidget(EmotionButton("Amusement", self, styleSheet="background-color: gray"))
        vbox1.addWidget(EmotionButton("Anger", self, styleSheet="background-color: gray"))
        vbox1.addWidget(EmotionButton("Awkwardness", self, styleSheet="background-color: gray"))
        vbox1.addWidget(EmotionButton("Boredom", self, styleSheet="background-color: gray"))
        vbox1.addWidget(EmotionButton("Confusion", self, styleSheet="background-color: gray"))


        vbox2 = QVBoxLayout()
        vbox2.addWidget(EmotionButton("Contempt", self, styleSheet="background-color: gray"))
        vbox2.addWidget(EmotionButton("Contentment", self, styleSheet="background-color: gray"))
        vbox2.addWidget(EmotionButton("Disappointment", self, styleSheet="background-color: gray"))
        vbox2.addWidget(EmotionButton("Disgust", self, styleSheet="background-color: gray"))
        vbox2.addWidget(EmotionButton("Embarrassment", self, styleSheet="background-color: gray"))


        vbox3 = QVBoxLayout()
        vbox3.addWidget(EmotionButton("Excitement", self, styleSheet="background-color: gray"))
        vbox3.addWidget(EmotionButton("Fear", self, styleSheet="background-color: gray"))
        vbox3.addWidget(EmotionButton("Guilt", self, styleSheet="background-color: gray"))
        vbox3.addWidget(EmotionButton("Happiness", self, styleSheet="background-color: gray"))
        vbox3.addWidget(EmotionButton("Jealousy", self, styleSheet="background-color: gray"))


        vbox4 = QVBoxLayout()
        vbox4.addWidget(EmotionButton("Love", self, styleSheet="background-color: gray"))
        vbox4.addWidget(EmotionButton("Nervousness", self, styleSheet="background-color: gray"))
        vbox4.addWidget(EmotionButton("Neutral", self, styleSheet="background-color: gray"))
        vbox4.addWidget(EmotionButton("Pain", self, styleSheet="background-color: gray"))
        vbox4.addWidget(EmotionButton("Pride", self, styleSheet="background-color: gray"))


        vbox5 = QVBoxLayout()
        vbox5.addWidget(EmotionButton("Relief", self, styleSheet="background-color: gray"))
        vbox5.addWidget(EmotionButton("Sadness", self, styleSheet="background-color: gray"))
        vbox5.addWidget(EmotionButton("Sexual Desire", self, styleSheet="background-color: gray"))
        vbox5.addWidget(EmotionButton("Surprise", self, styleSheet="background-color: gray"))
        vbox5.addWidget(EmotionButton("Tired", self, styleSheet="background-color: gray"))
        # vbox5.addSpacing(20)
        # vbox5.addWidget(EmotionButton("BAD CLIP", self))

        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        hbox.addLayout(vbox4)
        hbox.addLayout(vbox5)

        layout.addLayout(hbox)

        hbox2 = QGridLayout() #not a horizontal box

        self.intensity_bar = QSlider(Qt.Horizontal)
        self.intensity_bar.setMinimum(0)
        self.intensity_bar.setMaximum(5)
        self.intensity_bar.setTickInterval(1)
        self.intensity_bar.setTickPosition(3)
        hbox2.addWidget(QLabel("Intensity"), 0, 0, 1, 1)
        hbox2.addWidget(self.intensity_bar, 0, 1, 1, 5)

        self.confidence_bar = QSlider(Qt.Horizontal)
        self.confidence_bar.setMinimum(0)
        self.confidence_bar.setMaximum(5)
        self.confidence_bar.setTickInterval(1)
        self.confidence_bar.setTickPosition(3)
        hbox2.addWidget(QLabel("Confidence"), 1, 0, 1, 1)
        hbox2.addWidget(self.confidence_bar, 1, 1, 1, 5)

        hbox2.addWidget(QLabel("Not Set"), 2, 1, 1, 1)
        hbox2.addWidget(QLabel("Low"), 2, 2, 1, 1)
        hbox2.addWidget(QLabel("High"), 2, 5, 1, 1, Qt.AlignRight)

        hbox2.addWidget(self.submit_button, 0, 6, 1, 3)
        layout.addLayout(hbox2)

        # self.slider_bar = QSlider(Qt.Horizontal)
        # self.slider_bar.setMinimum(1)
        # self.slider_bar.setMaximum(5)
        # self.slider_bar.setTickInterval(1)
        # self.slider_bar.setTickPosition(3)
        # hbox2.addWidget(self.slider_bar, 0, 0, 1, 5)
        # hbox2.addWidget(QLabel("Low"), 1, 0, 1, 1)
        # hbox2.addWidget(QLabel("High"), 1, 4, 1, 1, Qt.AlignRight)
        #
        # hbox2.addWidget(self.submit_button, 0, 5, 1, 1)
        # layout.addLayout(hbox2)

        # layout.addWidget(EmotionButton("LOL"))

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(self.video_widget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

        # self.video_path = "videos/simple_test.mp4"
        # self.segments_path = "segment_labels/simple_test.csv"
        # self.segments = pd.read_csv(self.segments_path)

        # open the media and segments
        # needs an absolute path
        # self.mediaPlayer.setMedia(
        #     QMediaContent(QUrl.fromLocalFile(os.path.abspath(self.video_path))))
        # self.playButton.setEnabled(True)
        abspath = os.path.abspath("pilot_segments")
        files = os.listdir("pilot_segments")
        self.files = [abspath + "/" + x for x in files if ".mp4" in x]

        self.labels_path = "pilot_segments/labels.csv"
        self.create_or_load_csv()

        self.segmentLabel.setText(str(self.segment_counter) + "/" + str(len(self.files)-1))

        # SUPER SCUFFED approach, may not work depending on how long it takes to load
        # https://stackoverflow.com/questions/45175427/pyqt5-run-function-after-displaying-window
        QTimer.singleShot(1000, lambda: self.simpleplay(self.segment_counter))  # waits for this to finish until gui displayed


    def openFile(self):
        # fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
        #         QDir.homePath())
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        files = os.listdir(file)
        self.files = [file + "/" + x for x in files if ".mp4" in x]
        # print(fileName)
        print('saving current file')
        # self.segments.to_csv(self.segments_path, index=False)
        self.segment_counter = 0
        self.simpleplay(self.segment_counter)
        # if self.files[0] != '':
        #     self.mediaPlayer.setMedia(
        #             QMediaContent(QUrl.fromLocalFile(self.files[0])))
            # self.playButton.setEnabled(True)
        # self.segments_path = "segment_labels/" + os.path.basename(fileName).split('.')[0] + ".csv"
        # self.segments = pd.read_csv(self.segments_path)
        # self.segment_counter = 0
        # this is also pretty scuffed here
        # QTimer.singleShot(1000, lambda: self.simpleplay(self.segment_counter))  # waits for this to finish until gui displayed
        self.segmentLabel.setStyleSheet("color:black")
        self.segmentLabel.setText(str(self.segment_counter) + "/" + str(len(self.files) - 1))

    def saveFile(self):
        self.df.to_csv(self.labels_path)
        print("SUCCESSFULLY SAVED")

    def exitCall(self):
        print('exiting')
        # self.segments.to_csv(self.segments_path, index=False)
        sys.exit(app.exec_())

    def closeEvent(self, a):
        self.exitCall()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        # if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.SP_MediaPause))
        # else:
        #     self.playButton.setIcon(
        #             self.style().standardIcon(QStyle.SP_MediaPlay))
        pass

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

    def simpleplay(self, cnt):
        if self.files[cnt] != '':
            self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(self.files[cnt])))
            self.playButton.setEnabled(True)
        # self.segments_path = "segment_labels/" + os.path.basename(fileName).split('.')[0] + ".csv"
        # self.segments = pd.read_csv(self.segments_path)
        # self.segment_counter = 0
        # this is also pretty scuffed here
        QTimer.singleShot(1000, self.play_start_pause_end)
        # QTimer.singleShot(1000, lambda: self.mediaPlayer.play())

    def play_start_pause_end(self):
        self.mediaPlayer.setPosition(0)
        self.mediaPlayer.play()
        QTimer.singleShot(self.mediaPlayer.duration(), self.pause_after_playback)

    def pause_after_playback(self):
        self.mediaPlayer.pause()
        # easiest thing to do is to pause on the very first frame for people to draw on
        self.mediaPlayer.setPosition(0)

    # def playsegment(self, index):
    #     startframe = self.segments['start_frame'][index]
    #     endframe = self.segments['end_frame'][index]
    #     # arg is milliseconds (assuming video is run at 30 fps)
    #     frames_to_millisecs = startframe // 30 * 1000
    #     self.mediaPlayer.setPosition(frames_to_millisecs)
    #     self.mediaPlayer.play()
    #     while self.mediaPlayer.position() * 30 // 1000 < endframe:
    #         print(self.mediaPlayer.position() * 30 // 1000)
    #     self.mediaPlayer.pause()

    def create_or_load_csv(self):
        try:
            self.df = pd.read_csv(self.labels_path)
        except FileNotFoundError:
            self.df = pd.DataFrame({"file": [os.path.basename(x) for x in self.files],
                                    "label": ["UNLABELED",]*len(self.files),
                                    "intensity": [-1,]*len(self.files),
                                    "confidence": [-1, ] * len(self.files)
                                    })
            self.df.to_csv(self.labels_path)

        #logic to set segment_counter here so you can go back from where you left off
        # NOTE: this assumes there is at least 1 unlabeled.
        self.segment_counter = self.df.index[self.df.label == "UNLABELED"][0]
        print("Starting at index: " + str(self.segment_counter))

    def goback(self):
        if self.segment_counter <= 0:
            print("CANNOT GO BACK")
            return
        self.segment_counter = self.segment_counter - 1
        # self.playsegment(self.segment_counter)
        self.simpleplay(self.segment_counter)
        self.segmentLabel.setText(str(self.segment_counter) + "/" + str(len(self.files)-1))
        self.intensity_bar.setValue(0)
        self.confidence_bar.setValue(0)
        self.clear_drawing()

    def clear_drawing(self):
        self.drawingRef.hasBeenDrawnOn = False
        self.drawingRef.im.fill(Qt.transparent)
        self.drawingRef.update()

    def submit(self):
        print('submit pressed')
        global emotions_string_list
        if len(emotions_string_list) < 1:
            print('no emotions pressed')
            return
        if self.intensity_bar.value() == 0:
            print('no intesnity set')
            return
        if self.confidence_bar.value() == 0:
            print('no confidence set')
            return
        if self.drawingRef.hasBeenDrawnOn == False:
            print("need to draw on this")
            return

        GlobalObject().dispatchEvent("success")

        concatted = '_'.join(emotions_string_list)
        self.df['label'][self.segment_counter] = concatted
        print('submitting: ' + concatted)
        emotions_string_list = []

        # slider bar
        self.df['intensity'][self.segment_counter] = self.intensity_bar.value()
        self.df['confidence'][self.segment_counter] = self.confidence_bar.value()

        self.intensity_bar.setValue(0)
        self.confidence_bar.setValue(0)

        # process the overlay
        base_no_ext = os.path.splitext(os.path.basename(self.files[self.segment_counter]))[0]
        imgs_out = "image_masks"
        if not os.path.exists(imgs_out):
            os.mkdir(imgs_out)
        self.drawingRef.im.save(os.path.join(imgs_out, base_no_ext + "_mask.jpg"), "jpg")
        self.clear_drawing()

        # save to file after every iteration
        self.df.to_csv(self.labels_path)

        if self.segment_counter >= len(self.files) - 1:
            print("YOURE FINISHED.")
            self.segmentLabel.setStyleSheet("color:green")
            self.thing = DonePopup()
            self.thing.show()
            return
        self.segment_counter = self.segment_counter + 1
        self.segmentLabel.setText(
            str(self.segment_counter) + "/" + str(len(self.files) - 1))
        # self.vidWindow.playsegment(self.vidWindow.segment_counter)
        self.simpleplay(self.segment_counter)

    def showEvent(self, ev):
        print("POS" + str(self.mapToGlobal(self.video_widget.pos()).x()))
        self.drawingRef.move(self.mapToGlobal(self.video_widget.pos()))

    def moveEvent(self, ev):
        self.drawingRef.move(self.mapToGlobal(self.video_widget.pos()))

    # def on_click(self):
    #     print('button hit')
    #     startframe = self.segments['start_frame'][self.segment_counter]
    #     endframe = self.segments['end_frame'][self.segment_counter]
    #     # arg is milliseconds (assuming video is run at 30 fps)
    #     frames_to_millisecs = startframe // 30 * 1000
    #     self.mediaPlayer.setPosition(frames_to_millisecs)
    #     self.mediaPlayer.play()
    #     while self.mediaPlayer.position() * 30 // 1000 < endframe:
    #         print(self.mediaPlayer.position() * 30 // 1000)
    #     self.mediaPlayer.pause()

    # def mousePressEvent(self, e):
    #     self.points << e.pos()
    #     self.update()
    #
    # def paintEvent(self, ev):
    #     qp = QPainter(self.video_widget)
    #     qp.setRenderHint(QPainter.Antialiasing)
    #     pen = QPen(Qt.red, 5)
    #     brush = QBrush(Qt.red)
    #     qp.setPen(pen)
    #     qp.setBrush(brush)
    #     for i in range(self.points.count()):
    #         qp.drawEllipse(self.points.point(i), 5, 5)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 480)
    player.setFixedSize(QSize(800, 800)) # use a fixed size
    # player.show()

    playGeom = player.geometry()
    vidWidgGeom = player.video_widget.geometry()
    xy = player.mapToGlobal(QPoint(vidWidgGeom.x(), vidWidgGeom.y()))
    print(xy)

    # todo: these x and y's are wrong until initialization I think
    paintOnMe = Overlay(xy.x(), xy.y(), vidWidgGeom.width(), vidWidgGeom.height())
    # player.layout().addWidget(paintOnMe)
    paintOnMe.raise_()
    # player.show()


    player.drawingRef = paintOnMe
    player.show()

    playGeom = player.geometry()
    vidWidgGeom = player.video_widget.geometry()
    xy = player.mapToGlobal(QPoint(vidWidgGeom.x(), vidWidgGeom.y()))
    print(xy)
    paintOnMe.mainX = xy.x()
    paintOnMe.mainY = xy.y()
    paintOnMe.mainWid = vidWidgGeom.width()
    paintOnMe.mainHei = vidWidgGeom.height()
    paintOnMe.show()
    # app.exec_()
    # player.playsegment(1)
    # player.playsegment(0)
    sys.exit(app.exec_())