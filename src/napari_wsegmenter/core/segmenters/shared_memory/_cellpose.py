import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from _memory_manager import get_shared_array  # type: ignore

model, last_parameters = None, None


def segment(shared_image, shared_segmentation, parameters):

    print("Loading libraries...")
    import cellpose.io
    import cellpose.models

    global model, model_type, last_parameters

    if (
        last_parameters is None
        or last_parameters["model_type"] != parameters["model_type"]
        or last_parameters["use_gpu"] != parameters["use_gpu"]
    ):
        print("Loading model...")
        model_type = parameters["model_type"]
        model = cellpose.models.Cellpose(
            gpu=parameters["use_gpu"], model_type=model_type
        )
    last_parameters = parameters

    with get_shared_array(shared_image) as image:
        if model is None:
            return
        print("Computing segmentation...")
        masks, flows, styles, diams = model.eval(
            image,
            diameter=parameters["diameter"],
            channels=parameters["channels"],
        )
        print("segmentation finished.")
        with get_shared_array(shared_segmentation) as segmentation_array:
            segmentation_array[:] = masks
            return shared_segmentation
