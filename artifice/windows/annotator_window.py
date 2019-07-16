"""Create the annotator window for use by ann.py."""

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt


class AnnotatorWindow(QWidget):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.setWindowTitle("Annotator")


    
    
def main():
  """For testing."""
  app = QApplication([])
  window = AnnotatorWindow()
  window.show()
  app.exec_()

if __name__ == '__main__':
  main()
