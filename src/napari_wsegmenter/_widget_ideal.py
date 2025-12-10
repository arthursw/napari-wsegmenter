from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from magicgui import magic_factory
from napari import environemnt_manager
from napari.types import LayerDataTuple

if TYPE_CHECKING:
    import napari
    import napari.types

SEGMENTERS_PATH = Path(__file__).resolve().parent

env_cellpose = environemnt_manager.create(
    "Cellpose",
    {
        "python": "3.10",
        "pip": ["wetlands==0.4.4"],
        "conda": ["cellpose==3.1.0"],
    },
)
env_stardist = environemnt_manager.create(
    "StarDist",
    {
        "python": "3.10",
        "pip": [
            "wetlands==0.4.4",
            "tensorflow==2.16.1",
            "csbdeep==0.8.1",
            "stardist==0.9.1",
        ],
    },
)
env_sam = environemnt_manager.create(
    "SAM",
    {
        "python": "3.10",
        "pip": ["wetlands==0.4.4", "sam2==1.1.0", "huggingface_hub==0.29.2"],
    },
)
for env in [env_cellpose, env_stardist, env_sam]:
    env.launch()


@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = True,
    diameter: float = 30.0,
) -> LayerDataTuple:
    segmentation = np.zeros(image.shape, "uint8")
    env_cellpose.execute(
        SEGMENTERS_PATH / "_cellpose.py",
        "segment",
        (
            image,
            segmentation,
            {
                "model_type": model_type,
                "use_gpu": use_gpu,
                "diameter": diameter,
                "channels": [0, 0],
            },
        ),
    )
    return cast(
        LayerDataTuple,
        (segmentation, {"name": "Cellpose segmentation"}, "labels"),
    )


@magic_factory(
    model_name={"choices": ["2D_versatile_fluo", "2D_paper_dsb2018"]}
)
def stardist(
    image: "napari.types.ImageData", model_name="2D_versatile_fluo"
) -> LayerDataTuple:
    segmentation = np.zeros(image.shape, "uint8")
    env_stardist.execute(
        SEGMENTERS_PATH / "_stardist.py",
        "segment",
        (
            image,
            segmentation,
            {"model_name": model_name},
        ),
    )
    return cast(
        LayerDataTuple,
        (
            segmentation,
            {"name": "Stardist segmentation"},
            "labels",
        ),
    )


@magic_factory()
def sam(
    image: "napari.types.ImageData",
    use_gpu: bool = True,
    points_per_side: int = 8,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> LayerDataTuple:
    segmentation = np.zeros(image.shape, "uint8")
    env_sam.execute(
        SEGMENTERS_PATH / "_sam.py",
        "segment",
        (
            image,
            segmentation,
            {
                "use_gpu": use_gpu,
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
            },
        ),
    )
    return cast(
        LayerDataTuple,
        (segmentation, {"name": "SAM segmentation"}, "labels"),
    )


# Env and shared memory would be cleaned on plugin exit
