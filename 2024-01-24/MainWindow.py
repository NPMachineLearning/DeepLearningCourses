import sys
import time

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QListWidgetItem, QFileDialog

from DetectThread import DetectThread
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
        self.btnPath.clicked.connect(self.btnPath_clicked)
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
        btn.tag = pix.tag
        btn.clicked.connect(self.btn_clicked)
    def btnPath_clicked(self):
        path = QFileDialog.getExistingDirectory()
        if path != "":
            self.path = path.replace("\\", "/")
            self.lblPath.setText(self.path)
            self.listWidget.clear()
            self.pictureThread.runFlag = False
            self.pictureThread = PictureThread(self.path)
            self.pictureThread.callback.connect(self.pictureThreadCallback)
            self.pictureThread.start()
    def btn_clicked(self):
        btn = self.sender()
        self.lblStatus.setText("Analyzing ....")
        self.time1 = time.time()
        self.detectThread = DetectThread(self.model, btn.tag)
        self.detectThread.callback.connect(self.detectThreadCallback)
        self.detectThread.start()
    def detectThreadCallback(self, img):
        self.time2 = time.time()
        self.lblStatus.setText(f"Time: {self.time2 - self.time1:.5f} seconds")
        pix = QPixmap(
            QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        )
        pr = pix.width()/pix.height()
        lr = self.lblImg.width()/self.lblImg.height()
        print(pix.width(), pix.height())
        print(self.lblImg.width(), self.lblImg.height())
        if pr > lr:
            pix = pix.scaled(self.lblImg.width(), self.lblImg.height())
        else:
            pix = pix.scaled(self.lblImg.width(), self.lblImg.height())
        self.lblImg.setPixmap(pix)
    def closeEvent(self, event):
        if self.pictureThread is not None:
            self.pictureThread.runFlag = False

app = QApplication(sys.argv)
mainWindow = MainWindow()
mainWindow.showMaximized()
app.exec()