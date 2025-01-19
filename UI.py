import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QWidget, QGridLayout, QFileDialog, QVBoxLayout
)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import QSize

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cat image generator")
        self.setGeometry(100, 100, 500, 500)

        self.widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.widget)
        self.widget.setLayout(self.main_layout)
        self.setCentralWidget(self.widget)

        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)

        self.generate_button = QPushButton("Generate images", self)
        self.generate_button.setGeometry(100, 150, 200, 50)
        self.generate_button.clicked.connect(self.generate_new_images)
        self.main_layout.addWidget(self.generate_button)

    def generate_new_images(self):
        for i in reversed(range(self.grid_layout.count())):
            widget_to_remove = self.grid_layout.itemAt(i).widget()
            self.grid_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        if not os.path.exists('img'):
            os.makedirs('img')
        else:
            for file_name in os.listdir('img'):
                file_path = os.path.join('img', file_name)
                # os.remove(file_path)

        # Generate new images and save into 'img' folder in png/jpg/whatever

        images = os.listdir('img')
        images = images[:25]

        for i, img_path in enumerate(images):
            row = i // 5
            col = i % 5

            button = QPushButton(self)
            button.setIcon(QIcon(QPixmap('img/' + img_path)))
            button.setIconSize(QSize(100, 100))
            button.setFixedSize(120, 120)
            button.clicked.connect(lambda _, path=img_path: self.image_clicked(path))
            self.grid_layout.addWidget(button, row, col)

        self.generate_button.setText("Generate again")

    def image_clicked(self, path):
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            path,
            "Images (*.png *.jpg *.jpeg)"
        )
        if save_path:
            original_path = os.path.join('imgg', path)
            with open(original_path, 'rb') as src_file:
                with open(save_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())

app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec())
