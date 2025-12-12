import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from magicgui import magic_factory
from napari.types import LayerDataTuple
from qtpy.QtWidgets import QMessageBox

from wetlands.environment_manager import EnvironmentManager
from wetlands.ndarray import NDArray, update_ndarray

if TYPE_CHECKING:
    import napari
    import napari.types

PYTHON_VERSION = "3.10"
WETLANDS_VERSION = "wetlands==0.4.4"
SEGMENTERS_PATH = Path(__file__).resolve().parent

logging.getLogger("wetlands").addHandler(logging.StreamHandler())
logging.getLogger("wetlands").setLevel(logging.DEBUG)

# Create the Environment Manager
# This should be provided by Napari
# Use debug=True to be able to debug envs
# See https://arthursw.github.io/wetlands/latest/debugging/
environment_manager = EnvironmentManager(debug=True)

# Create the environments
env_cellpose = environment_manager.create(
    "Cellpose",
    {
        "python": PYTHON_VERSION,
        "pip": [WETLANDS_VERSION],  # Wetlands must be in the env dependencies
        #  when using NDArray
        "conda": ["cellpose==3.1.0"],
    },
)
env_stardist = environment_manager.create(
    "StarDist",
    {
        "python": PYTHON_VERSION,
        "pip": [
            WETLANDS_VERSION,
            "tensorflow==2.16.1",
            "csbdeep==0.8.1",
            "stardist==0.9.1",
        ],
    },
)
env_sam = environment_manager.create(
    "SAM",
    {
        "python": PYTHON_VERSION,
        "pip": [WETLANDS_VERSION, "sam2==1.1.0", "huggingface_hub==0.29.2"],
    },
)
# Launch the server in envs which will listen for execution orders
for env in [env_cellpose, env_stardist, env_sam]:
    env.launch()

# Global shared memory objects are used to avoid allocating shared memory when processing the same image multiple times
# The cellpose_simple() ignores this complexity and just allocate shared memory each time
shared_image: NDArray | None = None
# The shared segmentation is created on this side,
# but it could also be created in the environment
# as in the ndarray example https://arthursw.github.io/wetlands/latest/shared_memory/#ndarray-example
shared_segmentation: NDArray | None = None


# Helper to return a LayerDataTuple
# Copy the shared_memory since it will eventually be unlinked
def layer(
    ndarray: NDArray | None, name: str, layer_type: str = "labels"
) -> LayerDataTuple:
    if ndarray is None:
        raise Exception("NDArray is undefined.")
    return cast(
        LayerDataTuple, (ndarray.array.copy(), {"name": name}, layer_type)
    )


# Update the gloabl ndarrays from the image
# (create if size or dtype differs, update otherwise)
def update_shared_memory(image: "napari.types.ImageData"):
    global shared_image, shared_segmentation
    # Create or update the shared memory from the image data
    shared_image = update_ndarray(image, shared_image)
    # Create a shared memory for the segmentation if necessary
    shared_segmentation = update_ndarray(
        shape=image.shape[:2], dtype="uint8", ndarray=shared_segmentation
    )


# Computes the Cellpose segmentation using the global shared memory
@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = False,
    diameter: float = 30.0,
) -> LayerDataTuple:
    update_shared_memory(image)
    env_cellpose.execute(
        SEGMENTERS_PATH / "_cellpose.py",
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
    return layer(shared_segmentation, "Cellpose segmentation")


# Same as cellpose() but simpler because it creates a new shared memory each time,
# instead of the global vars.
# It uses the context manager which frees the shared memory on return.
@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose_local(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = False,
    diameter: float = 30.0,
) -> LayerDataTuple:
    with (
        NDArray(image) as ndimage,
        NDArray(shape=image.shape[:2], dtype="uint8") as ndsegmentation,
    ):
        env_cellpose.execute(
            SEGMENTERS_PATH / "_cellpose.py",
            "segment",
            (
                ndimage,
                ndsegmentation,
                {
                    "model_type": model_type,
                    "use_gpu": use_gpu,
                    "diameter": diameter,
                    "channels": [0, 0],
                },
            ),
        )
        return layer(ndsegmentation, "Cellpose segmentation")


# Same as cellpose_local() but even simpler because it does not use shared memory
# Instead, it pickles the image (sent with TCP socket)
# Note that it call segment_array(), not segment()
@magic_factory(model_type={"choices": ["cyto3", "cyto2", "nuclei"]})
def cellpose_pickle(
    image: "napari.types.ImageData",
    model_type="cyto3",
    use_gpu: bool = False,
    diameter: float = 30.0,
) -> LayerDataTuple:
    segmentation = env_cellpose.execute(
        SEGMENTERS_PATH / "_cellpose.py",
        "segment_array",
        (
            image,
            {
                "model_type": model_type,
                "use_gpu": use_gpu,
                "diameter": diameter,
                "channels": [0, 0],
            },
        ),
    )
    return cast(
        LayerDataTuple, (segmentation, {"name": "Segmentation"}, "labels")
    )


# Computes the StarDist segmentation using the global shared memory
@magic_factory(
    model_name={"choices": ["2D_versatile_fluo", "2D_paper_dsb2018"]}
)
def stardist(
    image: "napari.types.ImageData", model_name="2D_versatile_fluo"
) -> LayerDataTuple:
    update_shared_memory(image)
    env_stardist.execute(
        SEGMENTERS_PATH / "_stardist.py",
        "segment",
        (
            shared_image,
            shared_segmentation,
            {"model_name": model_name},
        ),
    )
    return layer(shared_segmentation, "Stardist segmentation")


# Computes the SAM segmentation using the global shared memory
@magic_factory()
def sam(
    image: "napari.types.ImageData",
    use_gpu: bool = False,
    points_per_side: int = 8,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
) -> LayerDataTuple:
    update_shared_memory(image)
    env_sam.execute(
        SEGMENTERS_PATH / "_sam.py",
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
    return layer(shared_segmentation, "SAM segmentation")


@magic_factory(
    call_button="Exit plugin",
)
def exit_button():
    exit_environments()
    QMessageBox.critical(
        None,
        "Warning",
        "All environments are exited and shared memory is unlinked. The plugin will not work anymore.",
    )


def exit_environments():
    if shared_image is not None:
        shared_image.dispose()
    if shared_segmentation is not None:
        shared_segmentation.dispose()
    environment_manager.exit()


# I need something like this
# widget.closed.connect(exit_environments)
