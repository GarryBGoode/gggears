from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSplitter,
    QFrame,
)
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
import gggears as gg
from ocp_vscode import show, set_port, Camera, set_defaults

# Only needed for access to command line arguments
import sys

import subprocess
import inspect
from collections.abc import Iterable


# subprocess.Popen(["python", "-m", "ocp_vscode"])

geartypes = [
    "SpurGear",
    "SpurRingGear",
    "HelicalGear",
    "HelicalRingGear",
    "BevelGear",
    "CycloidGear",
]


class ViewerWindow(QFrame):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gggears GUI Viewer")
        self.webview = QWebEngineView()
        self.webview.setUrl(QUrl("http://127.0.0.1:3838/viewer"))
        layout = QVBoxLayout()
        layout.addWidget(self.webview)
        self.setLayout(layout)


class InputArgPanel(QWidget):
    def __init__(self, signature: inspect.Signature):
        super().__init__()
        self.signature = signature
        self.dictionary = self.init_dict()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        keys = self.signature.parameters.keys()

        self.spin_boxes = {}
        for key in keys:
            label = QLabel(f"{key}:")

            value = self.signature.parameters[key].default
            if isinstance(value, Iterable):
                layout_loc = QHBoxLayout()
                for i in range(len(value)):
                    box = QDoubleSpinBox()
                    box.setValue(value[i])
                    box.valueChanged.connect(
                        lambda val, k=key, idx=i: self.update_dict(k[idx], val)
                    )
                    layout_loc.addWidget(box)
                input_widget = QWidget()
                input_widget.setLayout(layout_loc)
            else:
                if self.signature.parameters[key].annotation is int:
                    input_widget = QSpinBox()
                    input_widget.valueChanged.connect(
                        lambda val, k=key: self.update_dict(k, val)
                    )
                    if value != inspect.Parameter.empty:
                        input_widget.setValue(value)
                    if key == "number_of_teeth":
                        input_widget.setValue(12)
                elif self.signature.parameters[key].annotation is bool:
                    input_widget = QCheckBox()
                    input_widget.stateChanged.connect(
                        lambda val, k=key: self.update_dict(k, val)
                    )
                    input_widget.setChecked(value)
                elif self.signature.parameters[key].annotation is float:
                    input_widget = QDoubleSpinBox()
                    input_widget.valueChanged.connect(
                        lambda val, k=key: self.update_dict(k, val)
                    )
                    if value != inspect.Parameter.empty:
                        input_widget.setValue(value)

            layout.addWidget(label)
            layout.addWidget(input_widget)
            self.spin_boxes[key] = input_widget

        self.setLayout(layout)
        self.setWindowTitle("Dictionary Spin Boxes")
        self.show()

    def update_dict(self, key, value):
        self.dictionary[key] = value

    def init_dict(self):
        dictionary = {}
        for key in self.signature.parameters.keys():
            value = self.signature.parameters[key].default
            if isinstance(value, Iterable):
                dictionary[key] = value
            elif value != inspect.Parameter.empty:
                dictionary[key] = value
            else:
                dictionary[key] = 0

        # number of teeth hack
        dictionary["number_of_teeth"] = 12
        return dictionary


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gggears GUI Main")
        layout_base = QHBoxLayout()

        self.viewer = ViewerWindow()
        self.viewer.setFrameShape(QFrame.Shape.Box)
        self.viewer.setFrameShadow(QFrame.Shadow.Raised)

        self.gear_selector_left = QComboBox(self)
        self.gear_selector_left.addItems(geartypes)
        self.gear_selector_left.currentTextChanged.connect(self.change_gear_type_left)

        self.gear_selector_right = QComboBox(self)
        self.gear_selector_right.addItems(geartypes)
        self.gear_selector_right.currentTextChanged.connect(self.change_gear_type_right)

        generate_button_left = QPushButton("Generate1!")
        generate_button_left.clicked.connect(self.generate_gear_left)

        generate_button_right = QPushButton("Generate2!")
        generate_button_right.clicked.connect(self.generate_gear_right)

        cls = getattr(gg, geartypes[0])
        sig = inspect.signature(cls)
        sig2 = inspect.signature(cls)

        self.numpanel_left = InputArgPanel(sig)
        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.gear_selector_left)
        self.left_layout.addWidget(QLabel("Gear Parameters"))
        self.left_layout.addWidget(self.numpanel_left)
        self.left_layout.addWidget(generate_button_left)

        self.numpanel_right = InputArgPanel(sig2)
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.gear_selector_right)
        self.right_layout.addWidget(QLabel("Gear Parameters"))
        self.right_layout.addWidget(self.numpanel_right)
        self.right_layout.addWidget(generate_button_right)

        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.Box)
        panel.setFrameShadow(QFrame.Shadow.Raised)
        panel_layout = QHBoxLayout()
        panel_layout.addLayout(self.left_layout)
        panel_layout.addLayout(self.right_layout)
        panel.setLayout(panel_layout)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(panel)
        splitter.addWidget(self.viewer)
        layout_base.addWidget(splitter)

        widget = QWidget()
        widget.setLayout(layout_base)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)

        self.gearparts = [gg.Part(), gg.Part()]

    def generate_gear_left(self):

        sig = self.numpanel_left.signature
        dict = self.numpanel_left.dictionary
        positional_args = []
        keyword_args = {}
        for key in sig.parameters.keys():
            value = sig.parameters[key].default
            if isinstance(value, Iterable):
                keyword_args[key] = dict[key]
            elif value == inspect.Parameter.empty:
                positional_args.append(dict[key])
            else:
                keyword_args[key] = dict[key]

        cls = getattr(gg, self.gear_selector_left.currentText())

        self.gear1 = cls(*positional_args, **keyword_args)
        gearpart = self.gear1.build_part()
        gearpart.label = "Gear1"
        self.gearparts[0] = gearpart

        show(self.gearparts, reset_camera=Camera.KEEP)

    def generate_gear_right(self):

        sig = self.numpanel_right.signature
        dict = self.numpanel_right.dictionary
        positional_args = []
        keyword_args = {}
        for key in sig.parameters.keys():
            value = sig.parameters[key].default
            if isinstance(value, Iterable):
                keyword_args[key] = dict[key]
            elif value == inspect.Parameter.empty:
                positional_args.append(dict[key])
            else:
                keyword_args[key] = dict[key]

        cls = getattr(gg, self.gear_selector_right.currentText())
        self.gear2 = cls(*positional_args, **keyword_args)
        self.gear2.mesh_to(self.gear1)
        gearpart2 = self.gear2.build_part()
        gearpart2.label = "Gear2"
        self.gearparts[1] = gearpart2

        show(self.gearparts, reset_camera=Camera.KEEP)

    def change_gear_type_left(self, s):
        cls = getattr(gg, s)
        sig = inspect.signature(cls)
        index = self.left_layout.indexOf(self.numpanel_left)
        self.left_layout.removeWidget(self.numpanel_left)
        self.numpanel_left.deleteLater()
        self.numpanel_left = InputArgPanel(sig)
        self.left_layout.insertWidget(index, self.numpanel_left)

    def change_gear_type_right(self, s):
        cls = getattr(gg, s)
        sig = inspect.signature(cls)
        index = self.right_layout.indexOf(self.numpanel_right)
        self.right_layout.removeWidget(self.numpanel_right)
        self.numpanel_right.deleteLater()
        self.numpanel_right = InputArgPanel(sig)
        self.right_layout.insertWidget(index, self.numpanel_right)


ocp_view_process = subprocess.Popen(
    ["python", "-m", "ocp_vscode", "--port", "3838", "--theme", "dark"]
)
set_port(3838)
set_defaults(reset_camera=Camera.KEEP, grid=True)


dark_stylesheet = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
}

QPushButton {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
}

QPushButton:hover {
    background-color: #4c4c4c;
}

QLineEdit {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
}

QLabel {
    color: #ffffff;
}

QComboBox {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
}

QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #ffffff;
    selection-background-color: #4c4c4c;
}
"""


app = QApplication(sys.argv)
# Apply the stylesheet to your application
app.setStyleSheet(dark_stylesheet)
window = MainWindow()
window.show()
app.exec()

# make sure its dead?
ocp_view_process.kill()
