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
    QProgressBar,
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

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._task: Any = None

    def _set_content(self, form: QFormLayout, run_label: str) -> None:
        self.run_button = QPushButton(run_label)
        self.run_button.clicked.connect(self.run)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def run(self) -> None:
        raise NotImplementedError

    def _run_worker(
        self,
        parameters: dict[str, Any],
        *,
        command_id: str | None = None,
    ) -> None:
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            self.status_label.setText("Select an image layer first.")
            return

        self._set_busy(True)
        self.status_label.setText("Preparing plugin environment…")
        try:
            task = _execute_worker_command(
                command_id or self.COMMAND_ID,
                np.asarray(active_layer.data),
                parameters,
            )
        except (ImportError, KeyError, RuntimeError, ValueError) as error:
            self._on_error(error, notify=True)
            self._set_busy(False)
            return

        self._task = task
        task.add_progress_callback(self._on_progress)
        task.add_done_callback(self._on_done)

    def cancel(self) -> None:
        if self._task is None:
            return
        self.status_label.setText("Canceling…")
        self.cancel_button.setEnabled(False)
        self._task.cancel()

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.reset()

    def _on_progress(self, progress: Any) -> None:
        message = getattr(progress, "message", None)
        if message:
            self.status_label.setText(str(message))
        current = getattr(progress, "current", None)
        total = getattr(progress, "total", None)
        if total is None:
            total = getattr(progress, "maximum", None)
        if current is None or total is None:
            self.progress_bar.setRange(0, 0)
            return
        self.progress_bar.setRange(0, max(0, int(total)))
        self.progress_bar.setValue(max(0, int(current)))

    def _on_returned(self, labels: Any) -> None:
        if labels is None:
            return
        self.viewer.add_labels(np.asarray(labels), name=self.RESULT_NAME)
        self.status_label.setText("Segmentation complete.")

    def _on_error(self, error: Any, *, notify: bool = False) -> None:
        message = f"{self.RESULT_NAME} failed: {error}"
        self.status_label.setText(message)
        if notify:
            _show_error(message)

    def _on_done(self, task: Any) -> None:
        try:
            state = task.state.value
            if state == "completed":
                self._on_returned(task.result())
            elif state == "canceled":
                self.status_label.setText("Segmentation canceled.")
            else:
                self._on_error(task.error or "Unknown worker failure")
        finally:
            if self._task is task:
                self._task = None
                self._set_busy(False)


class CellposeWidget(BaseSegmenterWidget):
    COMMAND_ID = "napari-wsegmenter.cellpose_worker"
    RESULT_NAME = "Cellpose segmentation"

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__(napari_viewer)

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

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__(napari_viewer)

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

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__(napari_viewer)

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


class ThresholdWidget(BaseSegmenterWidget):
    """Small worker example for quickly exercising isolated environments."""

    RESULT_NAME = "Threshold segmentation"

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__(napari_viewer)

        self.environment = QComboBox()
        self.environment.addItem(
            "NumPy 1.26",
            "napari-wsegmenter.threshold_numpy1_worker",
        )
        self.environment.addItem(
            "NumPy 2.2",
            "napari-wsegmenter.threshold_numpy2_worker",
        )

        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(4)
        self.threshold.setRange(-1_000_000_000, 1_000_000_000)
        self.threshold.setValue(0.5)

        form = QFormLayout()
        form.addRow("Worker environment:", self.environment)
        form.addRow("Threshold:", self.threshold)
        self._set_content(form, "Run threshold")

    def run(self) -> None:
        self._run_worker(
            {"threshold": float(self.threshold.value())},
            command_id=str(self.environment.currentData()),
        )

    def _on_returned(self, result: Any) -> None:
        if result is None:
            return
        labels = np.asarray(result["labels"])
        self.viewer.add_labels(labels, name=self.RESULT_NAME)
        self.status_label.setText(
            f"Threshold complete in NumPy {result['numpy_version']}."
        )
