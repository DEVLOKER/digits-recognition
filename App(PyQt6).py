import sys, io
from PyQt6.QtWidgets import QWidget, QMainWindow, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QFont
from PyQt6.QtCore import Qt, QPoint, QSize
from Model import *
from PIL import ImageQt
from contextlib import redirect_stdout



class MainWindow(QMainWindow):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.setGeometry(100, 100, 1000, 500)
        self.setWindowTitle("Digit Recognizer")

        # results text
        results_label = QLabel("Thinking..") # font=("Helvetica", 48)
        predict_label = QLabel("")
        # put the canvas in the left widget
        canvas = PaintWidget(self.model, results_label, predict_label)
        # clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(canvas.clear)
        # recognize button
        recognize_button = QPushButton('Recognize')
        recognize_button.clicked.connect(canvas.classify_handwriting)

        # Left widget
        left_widget = QWidget()
        left_widget.setFixedSize(400,400)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(canvas)
        left_layout.addWidget(recognize_button)
        left_layout.addWidget(clear_button)
        left_widget.setLayout(left_layout)

        # Right widget
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        right_layout.addWidget(results_label)
        right_layout.addWidget(predict_label)

        
        # main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)


class PaintWidget(QWidget):
    def __init__(self, model, results_label, predict_label, width=400, height=400):
        super().__init__()
        self.model = model
        self.results_label = results_label
        self.predict_label = predict_label
        self.initUI()

    def initUI(self):
        self.setMouseTracking(True)
        self.points = []
        # model_summary = str(self.model.model.to_json())
        # model_summary = self.model.model.get_config()
        string_buffer = io.StringIO()
        with redirect_stdout(string_buffer):
            self.model.model.summary()
        model_summary = string_buffer.getvalue()#.replace('\t', '    ')
        # self.results_label.setFont(QFont('Arial', 10)) 
        self.results_label.setText(model_summary)
        
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
        painter.drawPoints(self.points)
        # for i in range(1, len(self.points)):
        #     x, y = self.points[i].x(), self.points[i].y()
        #     painter.drawPoint(x, y)
        #     # painter.drawLine(self.points[i - 1], self.points[i])
        #     # r = 8
        #     # painter.drawArc(x-r, y-r, r, r, 0, 360)

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
        digit, accuracy, prediction = self.model.predict_digit(img)
        self.results_label.setFont(QFont('Helvetica', 40))
        self.results_label.setText(' digit : {}% \n accuracy: {}%'.format(digit, int(accuracy*100)))
        predictions = ["{} => {:.2f}%".format(d, a*100) for d, a  in enumerate(prediction)]
        self.predict_label.setText("\n".join(predictions))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = Model(Model.model_name)
    # model = Model()
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())
