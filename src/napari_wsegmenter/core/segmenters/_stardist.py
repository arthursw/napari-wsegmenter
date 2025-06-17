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
        if (
            model_name == "2D_versatile_he"
            and len(image.shape) == 3
            and image.shape[2] != 3
        ):
            raise Exception(
                f"The 2D_versatile_he can only process RGB images ; image must have 3 channels (image.shape is {image.shape})."
            )
        if model_name == "2D_versatile_fluo" and (
            len(image.shape) != 2
            or len(image.shape) == 3
            and image.shape[2] != 1
        ):
            raise Exception(
                f"The 2D_versatile_fluo can only process grayscale images ; image must be 2D with a single channel (image.shape is {image.shape})."
            )
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
    image_normalized = normalize(image)
    labels, _ = model.predict_instances(image_normalized)
    return labels
