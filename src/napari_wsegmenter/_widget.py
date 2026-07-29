from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


def _execute_worker_command(command_id: str, *args: Any, **kwargs: Any) -> Any:
    from napari.plugins import execute_worker_command

    return execute_worker_command(command_id, *args, **kwargs)


def _show_error(message: str) -> None:
    from napari.utils.notifications import show_error

    show_error(message)


class BaseSegmenterWidget(QWidget):
    """Common host-side UI for an isolated segmentation worker."""

    COMMAND_ID = ""
    RESULT_NAME = "Segmentation"

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._task: Any = None

    def _set_content(self, form: QFormLayout, run_label: str) -> None:
        self.run_button = QPushButton(run_label)
        self.run_button.clicked.connect(self.run)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def run(self) -> None:
        raise NotImplementedError

    def _run_worker(self, parameters: dict[str, Any]) -> None:
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            self.status_label.setText("Select an image layer first.")
            return

        self._set_busy(True)
        self.status_label.setText("Preparing plugin environment…")
        try:
            task = _execute_worker_command(
                self.COMMAND_ID,
                np.asarray(active_layer.data),
                parameters,
            )
        except (ImportError, KeyError, RuntimeError, ValueError) as error:
            self._on_error(error)
            self._set_busy(False)
            return

        self._task = task
        task.events.started.connect(self._on_started)
        task.events.progress.connect(self._on_progress)
        task.events.returned.connect(self._on_returned)
        task.events.errored.connect(self._on_errored)
        task.events.canceled.connect(self._on_canceled)
        task.events.finished.connect(self._on_finished)

    def cancel(self) -> None:
        if self._task is None:
            return
        self.status_label.setText("Canceling…")
        self.cancel_button.setEnabled(False)
        self._task.cancel()

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _on_started(self, _event: Any) -> None:
        self.status_label.setText("Running segmentation…")

    def _on_progress(self, event: Any) -> None:
        progress = event.value
        message = getattr(progress, "message", None)
        if message:
            self.status_label.setText(str(message))

    def _on_returned(self, event: Any) -> None:
        labels = event.value
        if labels is None:
            return
        self.viewer.add_labels(np.asarray(labels), name=self.RESULT_NAME)
        self.status_label.setText("Segmentation complete.")

    def _on_errored(self, event: Any) -> None:
        self._on_error(event.value)

    def _on_error(self, error: Any) -> None:
        message = f"{self.RESULT_NAME} failed: {error}"
        self.status_label.setText(message)
        _show_error(message)

    def _on_canceled(self, _event: Any) -> None:
        self.status_label.setText("Segmentation canceled.")

    def _on_finished(self, _event: Any) -> None:
        self._task = None
        self._set_busy(False)


class CellposeWidget(BaseSegmenterWidget):
    COMMAND_ID = "napari-wsegmenter.cellpose_worker"
    RESULT_NAME = "Cellpose segmentation"

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__(viewer)

        self.model_type = QComboBox()
        self.model_type.addItems(["cyto3", "cyto2", "nuclei"])

        self.use_gpu = QCheckBox()

        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(0, 1000)
        self.diameter.setValue(30.0)

        form = QFormLayout()
        form.addRow("Model type:", self.model_type)
        form.addRow("Use GPU:", self.use_gpu)
        form.addRow("Diameter:", self.diameter)
        self._set_content(form, "Run Cellpose")

    def run(self) -> None:
        self._run_worker(
            {
                "model_type": self.model_type.currentText(),
                "use_gpu": self.use_gpu.isChecked(),
                "diameter": float(self.diameter.value()),
                "channels": [0, 0],
            }
        )


class StardistWidget(BaseSegmenterWidget):
    COMMAND_ID = "napari-wsegmenter.stardist_worker"
    RESULT_NAME = "StarDist segmentation"

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__(viewer)

        self.model_name = QComboBox()
        self.model_name.addItems(["2D_versatile_fluo", "2D_paper_dsb2018"])

        form = QFormLayout()
        form.addRow("Model:", self.model_name)
        self._set_content(form, "Run StarDist")

    def run(self) -> None:
        self._run_worker({"model_name": self.model_name.currentText()})


class SamWidget(BaseSegmenterWidget):
    COMMAND_ID = "napari-wsegmenter.sam_worker"
    RESULT_NAME = "SAM segmentation"

    def __init__(self, viewer: napari.Viewer) -> None:
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

        form = QFormLayout()
        form.addRow("Use GPU:", self.use_gpu)
        form.addRow("Points per side:", self.points_per_side)
        form.addRow("Pred IOU thresh:", self.pred_iou_thresh)
        form.addRow("Stability thresh:", self.stability_thresh)
        self._set_content(form, "Run SAM")

    def run(self) -> None:
        self._run_worker(
            {
                "use_gpu": self.use_gpu.isChecked(),
                "points_per_side": int(self.points_per_side.value()),
                "pred_iou_thresh": float(self.pred_iou_thresh.value()),
                "stability_score_thresh": float(self.stability_thresh.value()),
            }
        )
