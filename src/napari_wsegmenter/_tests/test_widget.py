from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from napari_wsegmenter import CellposeWidget, _widget


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def emit(self, value=None) -> None:
        event = SimpleNamespace(value=value)
        for callback in self.callbacks:
            callback(event)


class FakeTask:
    def __init__(self) -> None:
        self.events = SimpleNamespace(
            started=FakeSignal(),
            progress=FakeSignal(),
            returned=FakeSignal(),
            errored=FakeSignal(),
            canceled=FakeSignal(),
            finished=FakeSignal(),
        )
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


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

    task.events.progress.emit(
        SimpleNamespace(message="Installing Cellpose", current=1, maximum=2)
    )
    assert widget.status_label.text() == "Installing Cellpose"

    labels = np.ones(image.shape, dtype=np.int32)
    task.events.returned.emit(labels)
    task.events.finished.emit()

    np.testing.assert_array_equal(viewer.layers[-1].data, labels)
    assert viewer.layers[-1].name == "Cellpose segmentation"
    assert widget.status_label.text() == "Segmentation complete."
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()


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
    task.events.canceled.emit()
    task.events.finished.emit()

    assert task.cancel_calls == 1
    assert widget.status_label.text() == "Segmentation canceled."
    assert widget.run_button.isEnabled()


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

    task.events.errored.emit(RuntimeError("model download failed"))
    task.events.finished.emit()

    assert errors == ["Cellpose segmentation failed: model download failed"]
    assert "model download failed" in widget.status_label.text()


def test_widget_requires_an_active_layer(make_napari_viewer):
    widget = CellposeWidget(make_napari_viewer())

    widget.run()

    assert widget.status_label.text() == "Select an image layer first."
    assert widget.run_button.isEnabled()


def test_host_import_does_not_load_worker_dependencies():
    assert "wetlands" not in sys.modules
    assert "cellpose" not in sys.modules
    assert "stardist" not in sys.modules
    assert "sam2" not in sys.modules
