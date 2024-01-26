import sys
import time

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QApplication

from ClockThread import ClockThread
from ui.ui_mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thread test")
        self.setupUi(self)
        self.thread1 = None
        self.thread2 = None
        self.btn.clicked.connect(self.btn_clicked)
    def btn_clicked(self):
        if self.thread1 is not None:
            self.thread1.runFlag = False
        if self.thread2 is not None:
            self.thread2.runFlag = False
        self.thread1 = ClockThread(delay=10)
        self.thread1.callback.connect(self.thread1Callback)
        self.thread1.start()
        self.thread2 = ClockThread(delay=100)
        self.thread2.callback.connect(self.thread2Callback)
        self.thread2.start()
    def thread1Callback(self, i):
        self.lbl1.setText(str(i))
    def thread2Callback(self, i):
        self.lbl2.setText(str(i))

## bad
# class ClockThread(QThread):
#     def __init__(self, lbl, parent=None):
#         super().__init__(parent)
#         self.lbl = lbl
#     def run(self):
#         for i in range(100):
#             self.lbl.setText(str(i))
#             QThread.msleep(10)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()