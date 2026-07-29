from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class WorkerContext(Protocol):
    @property
    def cancel_requested(self) -> bool: ...

    def update(
        self,
        message: str,
        *,
        current: float | None = None,
        maximum: float | None = None,
    ) -> None: ...


def _update(
    context: WorkerContext | None,
    message: str,
    current: float,
    maximum: float,
) -> bool:
    if context is None:
        return False
    context.update(message, current=current, maximum=maximum)
    return context.cancel_requested


_cellpose_model: Any = None
_cellpose_model_key: tuple[str, bool] | None = None


def segment_cellpose(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext | None = None,
) -> np.ndarray | None:
    """Segment an image with Cellpose inside the managed environment."""
    global _cellpose_model, _cellpose_model_key

    if _update(napari_context, "Loading Cellpose", 0, 3):
        return None

    from cellpose import models

    model_key = (str(parameters["model_type"]), bool(parameters["use_gpu"]))
    if _cellpose_model is None or _cellpose_model_key != model_key:
        _cellpose_model = models.Cellpose(
            gpu=model_key[1],
            model_type=model_key[0],
        )
        _cellpose_model_key = model_key

    if _update(napari_context, "Running Cellpose", 1, 3):
        return None

    masks, *_ = _cellpose_model.eval(
        image,
        diameter=float(parameters["diameter"]),
        channels=list(parameters["channels"]),
    )
    if _update(napari_context, "Returning labels", 2, 3):
        return None
    return np.asarray(masks)


_stardist_model: Any = None
_stardist_model_name: str | None = None


def segment_stardist(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext | None = None,
) -> np.ndarray | None:
    """Segment an image with StarDist inside the managed environment."""
    global _stardist_model, _stardist_model_name

    if _update(napari_context, "Loading StarDist", 0, 3):
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

    if _update(napari_context, "Running StarDist", 1, 3):
        return None

    labels, _ = _stardist_model.predict_instances(normalize(worker_image))
    if _update(napari_context, "Returning labels", 2, 3):
        return None
    return np.asarray(labels)


_sam_predictor: Any = None
_sam_predictor_device: str | None = None
_sam_mask_generator: Any = None
_sam_generator_key: tuple[Any, ...] | None = None


def segment_sam(
    image: np.ndarray,
    parameters: dict[str, Any],
    *,
    napari_context: WorkerContext | None = None,
) -> np.ndarray | None:
    """Segment an image with SAM 2 inside the managed environment."""
    global _sam_generator_key
    global _sam_mask_generator
    global _sam_predictor
    global _sam_predictor_device

    if _update(napari_context, "Loading SAM 2", 0, 4):
        return None

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

    if _update(napari_context, "Running SAM 2", 1, 4):
        return None

    with (
        torch.inference_mode(),
        torch.autocast(
            device,
            dtype=torch.bfloat16,
        ),
    ):
        annotations = _sam_mask_generator.generate(np.asarray(image))

    if _update(napari_context, "Combining masks", 2, 4):
        return None

    labels = np.zeros(np.asarray(image).shape[:2], dtype=np.int32)
    for index, annotation in enumerate(annotations, start=1):
        if napari_context is not None and napari_context.cancel_requested:
            return None
        labels[np.asarray(annotation["segmentation"], dtype=bool)] = index

    if _update(napari_context, "Returning labels", 3, 4):
        return None
    return labels
