import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from _memory_manager import get_shared_array

model, last_parameters = None, None


def segment(shared_image, shared_segmentation, parameters):

    print("Loading libraries...")
    from csbdeep.utils import normalize

    global model, last_parameters

    if (
        last_parameters is None
        or last_parameters["model_name"] != parameters["model_name"]
    ):
        model_name = parameters["model_name"]
        if model_name.startswith("2D"):
            from stardist.models import StarDist2D

            model = StarDist2D.from_pretrained(model_name)
        else:
            from stardist.models import StarDist3D

            model = StarDist3D.from_pretrained(model_name)
    last_parameters = parameters
    if model is None:
        return

    print("Computing segmentation")
    with get_shared_array(shared_image) as image:
        labels, _ = model.predict_instances(normalize(image))
        with get_shared_array(shared_segmentation) as segmentation:
            segmentation[:] = labels
    return shared_segmentation
