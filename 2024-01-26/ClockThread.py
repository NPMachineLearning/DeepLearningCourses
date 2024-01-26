from PyQt5.QtCore import QThread, pyqtSignal


class ClockThread(QThread):
    callback = pyqtSignal(object)
    def __init__(self, delay=1000, parent=None):
        super().__init__(parent)
        self.delay = delay
        self.runFlag = True
    def run(self):
        i = 1
        while i<100 and self.runFlag:
            self.callback.emit(i)
            i+=1
            QThread.msleep(self.delay)
        # for i in range(100):
        #     self.callback.emit(i)
        #     QThread.msleep(self.delay)