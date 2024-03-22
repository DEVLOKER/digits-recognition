import sys
from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QPoint, QSize
from Model import *
from PIL import ImageQt


class MainWindow(QMainWindow):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.setGeometry(100, 100, 800, 500)
        self.setWindowTitle("Digit Recognizer")

        # Text edit
        label_results = QLabel("Thinking..") # font=("Helvetica", 48)
        # put the canvas in the left widget
        canvas = PaintWidget(self.model, label_results) # 400, 400)
        # clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(canvas.clear)
        # save button
        save_button = QPushButton('Recognize')
        save_button.clicked.connect(canvas.classify_handwriting)

        # Left widget
        left_widget = QWidget()
        left_widget.setFixedSize(400,400)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(canvas)
        left_widget.setLayout(left_layout)

        # Right widget
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        right_layout.addWidget(label_results)
        right_layout.addWidget(save_button)
        right_layout.addWidget(clear_button)
        
        # main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)


class PaintWidget(QWidget):
    def __init__(self, model=None, label=None, width=400, height=400):
        super().__init__()
        self.setMouseTracking(True)
        self.points = []
        self.model = model
        self.label = label
        
    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Set white background
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(self.rect())
        # Set black border
        painter.setPen(QPen(Qt.GlobalColor.white, 0, Qt.PenStyle.SolidLine))
        painter.drawRect(self.rect())
        # draw points
        painter.setPen(QPen(Qt.GlobalColor.black, 10, Qt.PenStyle.SolidLine))
        for i in range(1, len(self.points)):
            # painter.drawLine(self.points[i - 1], self.points[i])
            x, y = self.points[i].x(), self.points[i].y()
            painter.drawPoint(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # self.points = [event.pos()]
            self.points.append(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.MouseButton.LeftButton:
            self.points.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # self.points = []
            self.update()

    def to_image(self):
        # Create a QImage with the same size as the widget
        qimage = QImage(self.size(), QImage.Format.Format_RGB32)
        qimage.fill(255)  # Fill the image with white background
        # Render the widget onto the image
        painter = QPainter(qimage)
        self.render(painter)
        painter.end()
        # Save the image
        # image.save('drawn_image.png')  # Save the image to a file
        return ImageQt.fromqimage(qimage) # image

    def clear(self):
        self.points = []
        self.update()

    def classify_handwriting(self):
        img = self.to_image()
        digit, acc = self.model.predict_digit(img)
        self.label.setText(' digit : {} \n accuracy: {}%'.format(digit, int(acc*100)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = Model(Model.model_name)
    # model = Model()
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())
