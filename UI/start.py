import sys
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import *
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import *
import Main_Window  # module test. py

if __name__ == '__main__':
    app = QApplication(sys.argv)
    qp = QPixmap("./ICON/cover.png")
    qp.setDevicePixelRatio(2.0)
    splash = QSplashScreen(QPixmap(qp))
    splash.show()  # 展示启动图片
    app.processEvents()  # 防止进程卡死
    myMainWindow = QMainWindow()
    myUi = Main_Window.Ui_MainWindow()
    myUi.setupUi(myMainWindow)
    splash.finish(myMainWindow)
    myMainWindow.move(100, 100)
    myMainWindow.show()
    sys.exit(app.exec())
