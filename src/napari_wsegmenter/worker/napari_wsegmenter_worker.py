from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from napari.plugins import WorkerContext


def segment_threshold(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext,
) -> dict[str, Any] | None:
    """Threshold an image in either lightweight NumPy test environment."""

    napari_context.update("Thresholding image", current=0, maximum=2)
    if napari_context.cancel_requested:
        return None
    labels = (np.asarray(image) > float(parameters["threshold"])).astype(
        np.uint8
    )
    napari_context.update("Returning labels", current=1, maximum=2)
    if napari_context.cancel_requested:
        return None
    return {
        "labels": labels,
        "numpy_version": np.__version__,
        "threshold": float(parameters["threshold"]),
    }


_cellpose_model: Any = None
_cellpose_model_key: tuple[str, bool] | None = None


def segment_cellpose(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext,
) -> np.ndarray | None:
    """Segment an image with Cellpose inside the managed environment."""
    global _cellpose_model, _cellpose_model_key

    napari_context.update("Loading Cellpose", current=0, maximum=3)
    if napari_context.cancel_requested:
        return None

    from cellpose import models

    model_key = (str(parameters["model_type"]), bool(parameters["use_gpu"]))
    if _cellpose_model is None or _cellpose_model_key != model_key:
        _cellpose_model = models.Cellpose(
            gpu=model_key[1],
            model_type=model_key[0],
        )
        _cellpose_model_key = model_key

    napari_context.update("Running Cellpose", current=1, maximum=3)
    if napari_context.cancel_requested:
        return None

    masks, *_ = _cellpose_model.eval(
        image,
        diameter=float(parameters["diameter"]),
        channels=list(parameters["channels"]),
    )
    napari_context.update("Returning labels", current=2, maximum=3)
    if napari_context.cancel_requested:
        return None
    return np.asarray(masks)


_stardist_model: Any = None
_stardist_model_name: str | None = None


def segment_stardist(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext,
) -> np.ndarray | None:
    """Segment an image with StarDist inside the managed environment."""
    global _stardist_model, _stardist_model_name

    napari_context.update("Loading StarDist", current=0, maximum=3)
    if napari_context.cancel_requested:
        return None

    from csbdeep.utils import normalize

    model_name = str(parameters["model_name"])
    if _stardist_model is None or _stardist_model_name != model_name:
        if model_name.startswith("2D"):
            from stardist.models import StarDist2D

            _stardist_model = StarDist2D.from_pretrained(model_name)
        else:
            from stardist.models import StarDist3D

            _stardist_model = StarDist3D.from_pretrained(model_name)
        _stardist_model_name = model_name

    worker_image = np.asarray(image)
    if model_name == "2D_versatile_he" and (
        worker_image.ndim != 3 or worker_image.shape[-1] != 3
    ):
        raise ValueError(
            "The 2D_versatile_he model requires an RGB image with three channels."
        )
    if model_name == "2D_versatile_fluo" and worker_image.ndim == 3:
        if worker_image.shape[-1] == 1:
            worker_image = worker_image[..., 0]
        else:
            worker_image = worker_image.mean(axis=-1)

    napari_context.update("Running StarDist", current=1, maximum=3)
    if napari_context.cancel_requested:
        return None

    labels, _ = _stardist_model.predict_instances(normalize(worker_image))
    napari_context.update("Returning labels", current=2, maximum=3)
    if napari_context.cancel_requested:
        return None
    return np.asarray(labels)


_sam_predictor: Any = None
_sam_predictor_device: str | None = None
_sam_mask_generator: Any = None
_sam_generator_key: tuple[Any, ...] | None = None


def _sam_rgb_image(image: np.ndarray) -> np.ndarray:
    """Return an HWC RGB array accepted by SAM 2."""
    worker_image = np.asarray(image)
    if worker_image.ndim == 2:
        return np.repeat(worker_image[..., np.newaxis], 3, axis=-1)
    if worker_image.ndim != 3:
        raise ValueError("SAM 2 requires a 2D grayscale or RGB image.")
    if worker_image.shape[-1] == 1:
        return np.repeat(worker_image, 3, axis=-1)
    if worker_image.shape[-1] == 3:
        return worker_image
    if worker_image.shape[-1] == 4:
        return worker_image[..., :3]
    raise ValueError("SAM 2 requires a grayscale, RGB, or RGBA image.")


def segment_sam(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext,
) -> np.ndarray | None:
    """Segment an image with SAM 2 inside the managed environment."""
    global _sam_generator_key
    global _sam_mask_generator
    global _sam_predictor
    global _sam_predictor_device

    napari_context.update("Loading SAM 2", current=0, maximum=4)
    if napari_context.cancel_requested:
        return None

    worker_image = _sam_rgb_image(image)

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    device = "cuda" if bool(parameters["use_gpu"]) else "cpu"
    if _sam_predictor is None or _sam_predictor_device != device:
        _sam_predictor = build_sam2_hf(
            "facebook/sam2.1-hiera-large",
            device=torch.device(device),
            apply_postprocessing=False,
        )
        _sam_predictor_device = device
        _sam_mask_generator = None
        _sam_generator_key = None

    generator_key = (
        int(parameters["points_per_side"]),
        float(parameters["pred_iou_thresh"]),
        float(parameters["stability_score_thresh"]),
    )
    if _sam_mask_generator is None or _sam_generator_key != generator_key:
        _sam_mask_generator = SAM2AutomaticMaskGenerator(
            _sam_predictor,
            points_per_side=generator_key[0],
            pred_iou_thresh=generator_key[1],
            stability_score_thresh=generator_key[2],
        )
        _sam_generator_key = generator_key

    napari_context.update("Running SAM 2", current=1, maximum=4)
    if napari_context.cancel_requested:
        return None

    with (
        torch.inference_mode(),
        torch.autocast(
            device,
            dtype=torch.bfloat16,
        ),
    ):
        annotations = _sam_mask_generator.generate(worker_image)

    napari_context.update("Combining masks", current=2, maximum=4)
    if napari_context.cancel_requested:
        return None

    labels = np.zeros(worker_image.shape[:2], dtype=np.int32)
    for index, annotation in enumerate(annotations, start=1):
        if napari_context.cancel_requested:
            return None
        labels[np.asarray(annotation["segmentation"], dtype=bool)] = index

    napari_context.update("Returning labels", current=3, maximum=4)
    if napari_context.cancel_requested:
        return None
    return labels
