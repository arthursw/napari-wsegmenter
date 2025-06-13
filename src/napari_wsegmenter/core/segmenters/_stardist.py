import numpy as np

model, last_parameters = None, None


def segment(image, parameters) -> np.ndarray | None:

    print("Loading libraries...")
    from csbdeep.utils import normalize  # type: ignore

    global model, last_parameters

    if (
        last_parameters is None
        or last_parameters["model_name"] != parameters["model_name"]
    ):
        model_name = parameters["model_name"]
        if model_name.startswith("2D"):
            from stardist.models import StarDist2D  # type: ignore

            model = StarDist2D.from_pretrained(model_name)
        else:
            from stardist.models import StarDist3D  # type: ignore

            model = StarDist3D.from_pretrained(model_name)
    last_parameters = parameters
    if model is None:
        return

    print("Computing segmentation")
    labels, _ = model.predict_instances(normalize(image))
    return labels
