from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from napari._qt.qt_main_window import _instantiate_dock_widget

from napari_wsegmenter import (
    CellposeWidget,
    SamWidget,
    StardistWidget,
    ThresholdWidget,
    _widget,
)


class FakeTask:
    def __init__(self) -> None:
        self.state = SimpleNamespace(value="running")
        self.error = None
        self._result = None
        self._done_callbacks = []
        self._progress_callbacks = []
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1

    def add_progress_callback(self, callback) -> None:
        self._progress_callbacks.append(callback)

    def add_done_callback(self, callback) -> None:
        self._done_callbacks.append(callback)

    def progress(self, update) -> None:
        for callback in self._progress_callbacks:
            callback(update)

    def finish(self, state, *, result=None, error=None) -> None:
        self.state = SimpleNamespace(value=state)
        self._result = result
        self.error = error
        for callback in self._done_callbacks:
            callback(self)

    def result(self):
        return self._result


@pytest.mark.parametrize(
    "widget_class",
    [CellposeWidget, StardistWidget, SamWidget, ThresholdWidget],
)
def test_napari_injects_viewer_into_widget(widget_class, make_napari_viewer):
    viewer = make_napari_viewer()

    widget = _instantiate_dock_widget(widget_class, viewer)

    assert widget.viewer.layers is not None


def test_widget_runs_worker_and_adds_returned_labels(
    make_napari_viewer, monkeypatch
):
    viewer = make_napari_viewer()
    image = np.random.default_rng(0).random((12, 13))
    viewer.add_image(image)
    task = FakeTask()
    calls = []

    def execute(command_id, *args, **kwargs):
        calls.append((command_id, args, kwargs))
        return task

    monkeypatch.setattr(_widget, "_execute_worker_command", execute)
    widget = CellposeWidget(viewer)

    widget.run_button.click()

    assert calls[0][0] == "napari-wsegmenter.cellpose_worker"
    np.testing.assert_array_equal(calls[0][1][0], image)
    assert calls[0][1][1]["model_type"] == "cyto3"
    assert not widget.run_button.isEnabled()
    assert widget.cancel_button.isEnabled()
    assert not widget.progress_bar.isHidden()
    assert widget.progress_bar.minimum() == 0
    assert widget.progress_bar.maximum() == 0

    task.progress(
        SimpleNamespace(
            message="Installing Cellpose",
            phase=SimpleNamespace(value="provisioning"),
            current=1,
            total=2,
        )
    )
    assert widget.status_label.text() == "Installing Cellpose"
    assert widget.progress_bar.maximum() == 2
    assert widget.progress_bar.value() == 1

    task.progress(
        SimpleNamespace(
            message="Segmenting image",
            phase=SimpleNamespace(value="executing"),
            current=3,
            total=4,
        )
    )
    assert widget.status_label.text() == "Segmenting image"
    assert widget.progress_bar.maximum() == 4
    assert widget.progress_bar.value() == 3

    labels = np.ones(image.shape, dtype=np.int32)
    task.finish("completed", result=labels)

    np.testing.assert_array_equal(viewer.layers[-1].data, labels)
    assert viewer.layers[-1].name == "Cellpose segmentation"
    assert widget.status_label.text() == "Segmentation complete."
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    assert widget.progress_bar.isHidden()


def test_widget_cancels_active_task(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 4)))
    task = FakeTask()
    monkeypatch.setattr(
        _widget,
        "_execute_worker_command",
        lambda *args, **kwargs: task,
    )
    widget = CellposeWidget(viewer)
    widget.run()

    widget.cancel_button.click()
    task.finish("canceled")

    assert task.cancel_calls == 1
    assert widget.status_label.text() == "Segmentation canceled."
    assert widget.run_button.isEnabled()


def test_widget_explains_environment_lifecycle_cancellation(
    make_napari_viewer, monkeypatch
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 4)))
    task = FakeTask()
    monkeypatch.setattr(
        _widget,
        "_execute_worker_command",
        lambda *args, **kwargs: task,
    )
    widget = CellposeWidget(viewer)
    widget.run()

    task.finish(
        "canceled",
        error=RuntimeError(
            "The managed environment is being stopped or removed"
        ),
    )

    assert widget.status_label.text() == (
        "Segmentation canceled: The managed environment is being stopped "
        "or removed."
    )


def test_threshold_widget_selects_environment_and_handles_nested_result(
    make_napari_viewer, monkeypatch
):
    viewer = make_napari_viewer()
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    viewer.add_image(image)
    task = FakeTask()
    calls = []
    monkeypatch.setattr(
        _widget,
        "_execute_worker_command",
        lambda command_id, *args, **kwargs: (
            calls.append((command_id, args, kwargs)) or task
        ),
    )
    widget = ThresholdWidget(viewer)
    widget.environment.setCurrentIndex(1)
    widget.threshold.setValue(1.5)

    widget.run()

    assert calls[0][0] == "napari-wsegmenter.threshold_numpy2_worker"
    assert calls[0][1][1] == {"threshold": 1.5}
    labels = (image > 1.5).astype(np.uint8)
    task.finish(
        "completed",
        result={
            "labels": labels,
            "numpy_version": "2.2.6",
            "threshold": 1.5,
        },
    )

    np.testing.assert_array_equal(viewer.layers[-1].data, labels)
    assert widget.status_label.text() == "Threshold complete in NumPy 2.2.6."


def test_widget_presents_worker_failure(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 4)))
    task = FakeTask()
    errors = []
    monkeypatch.setattr(
        _widget,
        "_execute_worker_command",
        lambda *args, **kwargs: task,
    )
    monkeypatch.setattr(_widget, "_show_error", errors.append)
    widget = CellposeWidget(viewer)
    widget.run()

    task.finish("failed", error=RuntimeError("model download failed"))

    assert errors == []
    assert "model download failed" in widget.status_label.text()


def test_widget_presents_submission_failure(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 4)))
    errors = []

    def fail(*args, **kwargs):
        raise RuntimeError("worker command is unavailable")

    monkeypatch.setattr(_widget, "_execute_worker_command", fail)
    monkeypatch.setattr(_widget, "_show_error", errors.append)
    widget = CellposeWidget(viewer)

    widget.run()

    assert errors == [
        "Cellpose segmentation failed: worker command is unavailable"
    ]
    assert widget.run_button.isEnabled()
    assert "worker command is unavailable" in widget.status_label.text()


def test_widget_requires_an_active_layer(make_napari_viewer):
    widget = CellposeWidget(make_napari_viewer())

    widget.run()

    assert widget.status_label.text() == "Select an image layer first."
    assert widget.run_button.isEnabled()


def test_widget_handles_task_completed_before_callbacks_are_added(
    make_napari_viewer, monkeypatch
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 4)))
    labels = np.ones((4, 4), dtype=np.int32)

    class CompletedTask(FakeTask):
        def __init__(self):
            super().__init__()
            self.state = SimpleNamespace(value="completed")
            self._result = labels

        def add_done_callback(self, callback) -> None:
            callback(self)

    monkeypatch.setattr(
        _widget,
        "_execute_worker_command",
        lambda *args, **kwargs: CompletedTask(),
    )
    widget = CellposeWidget(viewer)

    widget.run()

    np.testing.assert_array_equal(viewer.layers[-1].data, labels)
    assert widget.run_button.isEnabled()


def test_host_import_does_not_load_worker_dependencies():
    assert "wetlands" not in sys.modules
    assert "cellpose" not in sys.modules
    assert "stardist" not in sys.modules
    assert "sam2" not in sys.modules
