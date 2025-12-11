from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari
    import napari.viewer

from wetlands.environment_manager import EnvironmentManager
from wetlands.ndarray import update_ndarray

SEGMENTERS_PATH = Path(__file__).resolve().parent


# ===============================================================
#                    BASE SEGMENTER WIDGET
# ===============================================================


class BaseSegmenterWidget(QWidget):
    """
    A base widget that handles:
    - creating an EnvironmentManager and one environment
    - allocating and updating shared memory
    - running a segmentation script
    """

    ENV_NAME = ""  # overridden in subclass
    ENV_SPEC = None  # overridden in subclass
    SCRIPT_NAME = ""  # overridden in subclass

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # --- Environment creation ---
        self.environment_manager = EnvironmentManager(debug=True)
        self.env = self.environment_manager.create(
            self.ENV_NAME, self.ENV_SPEC
        )
        self.env.launch()

        # --- Shared memory ---
        self.shared_image = None
        self.shared_segmentation = None

    # -------------------------------
    # Shared memory allocation/update
    # -------------------------------
    def update_shared_memory(self, image):
        if image is None:
            return
        self.shared_image = update_ndarray(image, self.shared_image)
        self.shared_segmentation = update_ndarray(
            shape=image.shape[:2],
            dtype="uint8",
            ndarray=self.shared_segmentation,
        )

    # -------------------------------
    # Running inside environment
    # -------------------------------
    def run_environment(self, args_dict):
        layer0 = self.viewer.layers.selection.active
        if layer0 is None:
            print("No active layer selected.")
            return

        self.update_shared_memory(layer0.data)

        self.env.execute(
            SEGMENTERS_PATH / self.SCRIPT_NAME,
            "segment",
            (
                self.shared_image,
                self.shared_segmentation,
                args_dict,
            ),
        )

        if self.shared_segmentation:
            self.viewer.add_labels(
                self.shared_segmentation.array,
                name=f"{self.ENV_NAME} segmentation",
            )

    # -------------------------------
    # Cleanup on widget close
    # -------------------------------
    def closeEvent(self, a0):
        if self.shared_image is not None:
            self.shared_image.dispose()
        if self.shared_segmentation is not None:
            self.shared_segmentation.dispose()
        self.environment_manager.exit()
        if a0:
            a0.accept()


# ===============================================================
#                        CELLPOSE WIDGET
# ===============================================================


class CellposeWidget(BaseSegmenterWidget):
    ENV_NAME = "Cellpose"
    ENV_SPEC = {
        "python": "3.10",
        "pip": ["wetlands==0.4.4"],
        "conda": ["cellpose==3.1.0"],
    }
    SCRIPT_NAME = "_cellpose.py"

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        # UI
        self.model_type = QComboBox()
        self.model_type.addItems(["cyto3", "cyto2", "nuclei"])

        self.use_gpu = QCheckBox()

        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(30.0)

        self.run_button = QPushButton("Run Cellpose")
        self.run_button.clicked.connect(self.run)

        # Layout
        form = QFormLayout()
        form.addRow("Model type:", self.model_type)
        form.addRow("Use GPU:", self.use_gpu)
        form.addRow("Diameter:", self.diameter)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def run(self):
        self.run_environment(
            {
                "model_type": self.model_type.currentText(),
                "use_gpu": self.use_gpu.isChecked(),
                "diameter": float(self.diameter.value()),
                "channels": [0, 0],
            }
        )


# ===============================================================
#                        STARDIST WIDGET
# ===============================================================


class StardistWidget(BaseSegmenterWidget):
    ENV_NAME = "StarDist"
    ENV_SPEC = {
        "python": "3.10",
        "pip": [
            "wetlands==0.4.4",
            "tensorflow==2.16.1",
            "csbdeep==0.8.1",
            "stardist==0.9.1",
        ],
    }
    SCRIPT_NAME = "_stardist.py"

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        self.model_name = QComboBox()
        self.model_name.addItems(["2D_versatile_fluo", "2D_paper_dsb2018"])

        self.run_button = QPushButton("Run Stardist")
        self.run_button.clicked.connect(self.run)

        form = QFormLayout()
        form.addRow("Model:", self.model_name)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def run(self):
        self.run_environment({"model_name": self.model_name.currentText()})


# ===============================================================
#                           SAM WIDGET
# ===============================================================


class SamWidget(BaseSegmenterWidget):
    ENV_NAME = "SAM"
    ENV_SPEC = {
        "python": "3.10",
        "pip": ["wetlands==0.4.4", "sam2==1.1.0", "huggingface_hub==0.29.2"],
    }
    SCRIPT_NAME = "_sam.py"

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        self.use_gpu = QCheckBox()

        self.points_per_side = QSpinBox()
        self.points_per_side.setRange(1, 2048)
        self.points_per_side.setValue(8)

        self.pred_iou_thresh = QDoubleSpinBox()
        self.pred_iou_thresh.setRange(0, 1)
        self.pred_iou_thresh.setValue(0.88)

        self.stability_thresh = QDoubleSpinBox()
        self.stability_thresh.setRange(0, 1)
        self.stability_thresh.setValue(0.95)

        self.run_button = QPushButton("Run SAM")
        self.run_button.clicked.connect(self.run)

        form = QFormLayout()
        form.addRow("Use GPU:", self.use_gpu)
        form.addRow("Points per side:", self.points_per_side)
        form.addRow("Pred IOU thresh:", self.pred_iou_thresh)
        form.addRow("Stability thresh:", self.stability_thresh)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def run(self):
        self.run_environment(
            {
                "use_gpu": self.use_gpu.isChecked(),
                "points_per_side": int(self.points_per_side.value()),
                "pred_iou_thresh": float(self.pred_iou_thresh.value()),
                "stability_score_thresh": float(self.stability_thresh.value()),
            }
        )
