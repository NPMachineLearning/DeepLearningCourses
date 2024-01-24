import sys

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QListWidgetItem

from ModelThread import ModelThread
from PictureThread import PictureThread
from ui.ui_mainwindow import Ui_MainWindow
import sys


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.path = "D:/DeepLearningCourses/2024-01-24/photos"
        self.lblPath.setText(self.path)
        self.lblStatus.setText("Loading the model ....")
        self.modelThread = ModelThread()
        self.modelThread.callback.connect(self.modelThreadCallback)
        self.modelThread.start()
    def modelThreadCallback(self, model):
        self.lblStatus.setText("Loading model successful!")
        self.model = model
        self.pictureThread = PictureThread(path=self.path)
        self.pictureThread.callback.connect(self.pictureThreadCallback)
        self.pictureThread.start()
    def pictureThreadCallback(self, pix):
        btn = QPushButton()
        btn.setIcon(QIcon(pix))
        btn.setIconSize(QSize(400, 300))
        item = QListWidgetItem()
        item.setSizeHint(QSize(400, 300))
        self.listWidget.addItem(item)
        self.listWidget.setItemWidget(item, btn)


app = QApplication(sys.argv)
mainWindow = MainWindow()
mainWindow.showMaximized()
app.exec()