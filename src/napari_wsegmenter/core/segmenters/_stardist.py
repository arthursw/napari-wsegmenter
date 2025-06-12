import numpy as np

model, last_parameters = None, None


def segment(image_path, segmentation_path, parameters):

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
    image = np.load(image_path)
    labels, _ = model.predict_instances(normalize(image))
    np.save(labels, segmentation_path)
    return segmentation_path
