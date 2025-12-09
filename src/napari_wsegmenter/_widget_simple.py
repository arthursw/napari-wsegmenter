from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui

from wetlands.environment_manager import (  # type: ignore
    EnvironmentManager,  # type: ignore
)
from wetlands.ndarray import NDArray, initialize_ndarray

if TYPE_CHECKING:
    import napari
    import napari.types

environemnt_manager = EnvironmentManager()
env_cellpose = environemnt_manager.create(
    "Cellpose", {"python": "3.10", "conda": ["cellpose==3.1.0"]}
)
env_stardist = environemnt_manager.create(
    "StarDist",
    {
        "python": "3.10",
        "pip": ["tensorflow==2.16.1", "csbdeep==0.8.1", "stardist==0.9.1"],
    },
)
env_sam = environemnt_manager.create(
    "SAM",
    {"python": "3.10", "conda": ["sam2==1.1.0", "huggingface_hub==0.29.2"]},
)
for env in [env_cellpose, env_stardist, env_sam]:
    env.launch()

shared_image: NDArray | None = None
shared_segmentation: NDArray | None = None


def initialize_images(image: "napari.types.ImageData"):
    global shared_image, shared_segmentation
    shared_image = initialize_ndarray(image.data, shared_image)
    if shared_image is None:
        return None
    shared_segmentation = initialize_ndarray(
        np.zeros(shared_image.array.shape, "uint8"), shared_segmentation
    )


@magicgui(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose(
    image: "napari.types.ImageData",
    model_type: str = "cyto3",
    use_gpu: bool = True,
    diameter: float = 30.0,
) -> None | np.ndarray | "napari.types.LabelsData":
    global shared_image, shared_segmentation
    initialize_images(image)
    env_cellpose.execute(
        "segmenters._cellpose.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {
                "model_type": model_type,
                "use_gpu": use_gpu,
                "diameter": diameter,
                "channels": [0, 0],
            },
        ),
    )
    return shared_segmentation.array


@magicgui(model_name={"choices": ["2D_versatile_fluo", "2D_paper_dsb2018"]})
def stardist(
    image: "napari.types.ImageData", model_name: str = "2D_versatile_fluo"
) -> None | np.ndarray | "napari.types.LabelsData":
    global shared_image, shared_segmentation
    initialize_images(image)
    env_stardist.execute(
        "segmenters._stardist.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {"model_name": model_name},
        ),
    )
    return shared_segmentation.array


@magicgui()
def sam(
    image: "napari.types.ImageData",
    use_gpu: bool = True,
    points_per_side: int = 8,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> None | np.ndarray | "napari.types.LabelsData":
    global shared_image, shared_segmentation
    initialize_images(image)
    env_sam.execute(
        "segmenters._sam.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {
                "use_gpu": use_gpu,
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
            },
        ),
    )
    return shared_segmentation.array


@magicgui(
    call_button="Exit plugin",
)
def exit_button():
    exit_environments()


def exit_environments():
    shared_image.dispose()
    shared_segmentation.dispose()
    environemnt_manager.exit()


# I need something like this
# segmenter_widget.closed.connect(exit_environments)
