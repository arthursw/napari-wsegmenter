from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

WORKER_SOURCE = Path(__file__).parents[1] / "worker"
sys.path.insert(0, str(WORKER_SOURCE))

import napari_wsegmenter_worker as worker_module  # noqa: E402

segment_cellpose = worker_module.segment_cellpose
segment_sam = worker_module.segment_sam
segment_stardist = worker_module.segment_stardist


class FakeContext:
    def __init__(self, *, canceled: bool = False) -> None:
        self.cancel_requested = canceled
        self.updates: list[tuple[str, int | None, int | None]] = []

    def update(
        self,
        message: str,
        *,
        current: int | None = None,
        maximum: int | None = None,
    ) -> None:
        self.updates.append((message, current, maximum))


def test_cellpose_worker_uses_arrays_and_reports_progress(monkeypatch):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)
    calls = []

    class FakeCellpose:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def eval(self, value, **kwargs):
            calls.append(("eval", value, kwargs))
            return expected, None, None, None

    monkeypatch.setitem(
        sys.modules,
        "cellpose",
        SimpleNamespace(models=SimpleNamespace(Cellpose=FakeCellpose)),
    )
    monkeypatch.setattr(worker_module, "_cellpose_model", None)
    monkeypatch.setattr(worker_module, "_cellpose_model_key", None)
    context = FakeContext()

    result = segment_cellpose(
        image,
        {
            "model_type": "cyto3",
            "use_gpu": False,
            "diameter": 30,
            "channels": [0, 0],
        },
        napari_context=context,
    )

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(calls[1][1], image)
    assert [update[0] for update in context.updates] == [
        "Loading Cellpose",
        "Running Cellpose",
        "Returning labels",
    ]


def test_stardist_worker_keeps_heavy_imports_inside_call(monkeypatch):
    expected = np.ones((3, 3), dtype=np.int32)

    class FakeStarDist:
        @classmethod
        def from_pretrained(cls, name):
            assert name == "2D_versatile_fluo"
            return cls()

        def predict_instances(self, image):
            assert image.shape == (3, 3)
            return expected, {}

    csbdeep = ModuleType("csbdeep")
    csbdeep_utils = ModuleType("csbdeep.utils")
    csbdeep_utils.normalize = lambda image: image
    stardist = ModuleType("stardist")
    stardist_models = ModuleType("stardist.models")
    stardist_models.StarDist2D = FakeStarDist
    monkeypatch.setitem(sys.modules, "csbdeep", csbdeep)
    monkeypatch.setitem(sys.modules, "csbdeep.utils", csbdeep_utils)
    monkeypatch.setitem(sys.modules, "stardist", stardist)
    monkeypatch.setitem(sys.modules, "stardist.models", stardist_models)
    monkeypatch.setattr(worker_module, "_stardist_model", None)
    monkeypatch.setattr(worker_module, "_stardist_model_name", None)

    result = segment_stardist(
        np.ones((3, 3, 3), dtype=np.float32),
        {"model_name": "2D_versatile_fluo"},
        napari_context=FakeContext(),
    )

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("image", "expected_image"),
    [
        (
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ),
        (
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
        ),
    ],
)
def test_sam_worker_returns_labels(monkeypatch, image, expected_image):
    first_mask = np.zeros((4, 4), dtype=bool)
    first_mask[:2, :2] = True
    second_mask = np.zeros((4, 4), dtype=bool)
    second_mask[3, 3] = True

    class FakeMaskGenerator:
        def __init__(self, predictor, **kwargs):
            assert predictor == "predictor"
            assert kwargs["points_per_side"] == 8

        def generate(self, value):
            np.testing.assert_array_equal(value, expected_image)
            return [
                {"segmentation": first_mask},
                {"segmentation": second_mask},
            ]

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.device = lambda value: value
    fake_torch.inference_mode = contextlib.nullcontext
    fake_torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
    automatic_mask_generator = ModuleType("sam2.automatic_mask_generator")
    automatic_mask_generator.SAM2AutomaticMaskGenerator = FakeMaskGenerator
    build_sam = ModuleType("sam2.build_sam")
    build_sam.build_sam2_hf = lambda *args, **kwargs: "predictor"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "sam2.automatic_mask_generator",
        automatic_mask_generator,
    )
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_sam)
    monkeypatch.setattr(worker_module, "_sam_predictor", None)
    monkeypatch.setattr(worker_module, "_sam_predictor_device", None)
    monkeypatch.setattr(worker_module, "_sam_mask_generator", None)
    monkeypatch.setattr(worker_module, "_sam_generator_key", None)

    result = segment_sam(
        image,
        {
            "use_gpu": False,
            "points_per_side": 8,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
        },
        napari_context=FakeContext(),
    )

    assert result is not None
    np.testing.assert_array_equal(result[:2, :2], 1)
    assert result[3, 3] == 2


def test_sam_worker_rejects_non_image_array():
    with pytest.raises(ValueError, match="2D grayscale or RGB"):
        segment_sam(
            np.zeros((2, 3, 4, 5), dtype=np.uint8),
            {},
            napari_context=FakeContext(),
        )


def test_worker_stops_before_heavy_import_when_canceled():
    result = segment_cellpose(
        np.zeros((2, 2)),
        {},
        napari_context=FakeContext(canceled=True),
    )

    assert result is None
