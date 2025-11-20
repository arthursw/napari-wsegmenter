"""
This is what _widget.py would look like with an ideal Wetlands

All environments are exited and shared memory freed when the widget is closed
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from magicgui import magicgui

from wetlands.environment_manager import (  # type: ignore
    EnvironmentManager,
    environment,  # type: ignore
)

if TYPE_CHECKING:
    import napari
    import napari.types

# Wetlands is initialized given the environments object which describes the environments and their dependencies
# Environments can be defined with a dict or a yml file
# The "transfer" object describes how parameters are sent to the other process:
# - "pickle": transfer any picklable object (default)
# - "shared_memory_numpy": transfer a numpy array using shared memory
# - "imageio": transfer an image using a temporary file and imageio
# - ...
EnvironmentManager.initialize(  # type: ignore
    {  # type: ignore          Use Wetlands.initialize instead of EnvironmentManager.initialize?
        "environments": {
            "Cellpose": {
                "dependencies": {
                    "python": "3.10",
                    "conda": ["cellpose==3.1.0"],
                },
            },
            "StarDist": "stardist/pyproject.toml",
            "SAM": "sam/environment.yml",
        },
    }
)


# The environment decorator enables to execute the decorated function in the given environment
# The transfer argument defines how the parameters are sent to the process
@environment(name="StarDist", transfer={"image": "imageio", "parameters": "pickle"})  # type: ignore
def stardist(Global, image, parameters):

    print("Loading libraries...")
    from csbdeep.utils import normalize  # type: ignore

    if (
        Global.last_parameters is None
        or Global.last_parameters["model_name"] != parameters["model_name"]
    ):
        model_name = parameters["model_name"]
        if model_name.startswith("2D"):
            from stardist.models import StarDist2D  # type: ignore

            Global.model = StarDist2D.from_pretrained(model_name)
        else:
            from stardist.models import StarDist3D  # type: ignore

            Global.model = StarDist3D.from_pretrained(model_name)
    Global.last_parameters = parameters
    if Global.model is None:
        return

    print("Computing segmentation")
    labels, _ = Global.model.predict_instances(normalize(image))
    return labels


# Decorated functions can import and use other modules
@environment(name="Cellpose", transfer={"image": "shared_memory_numpy"})
def cellpose(Global, image, parameters):
    sys.path.append(str(Path(__file__).parent))
    import segmenters._cellpose  # type: ignore

    return segmenters._cellpose.segment(image, parameters)


@magicgui(
    segmenter={"choices": ["stardist", "cellpose", "sam"]},
)
def segmenter_widget(
    image: "napari.types.ImageData",
    segmenter: "str",
) -> None | np.ndarray | "napari.types.LabelsData":

    # We can now make use of our decorated functions and commands
    match segmenter:
        case "stardist":
            return stardist(image, {"model_name": "2D_versatile_fluo"})  # type: ignore
        case "cellpose":
            return cellpose(
                image,
                {  # type: ignore
                    "model_type": "cyto3",
                    "use_gpu": False,
                    "diameter": 30.0,
                    "channels": [0, 0],
                },
            )
        case "sam":

            return EnvironmentManager.environments["sam"].execute(  # type: ignore
                "segmenters._cellpose.segment",
                {
                    "image": image,
                    "use_gpu": False,
                    "points_per_side": 8,
                    "pred_iou_thresh": 0.88,
                    "stability_score_thresh": 0.95,
                },
                transfert={"image": "shared_memory_numpy"},
            )

        case _:
            raise Exception("Unknown segmenter")


# All environments are exited and shared memory freed when the widget is closed
