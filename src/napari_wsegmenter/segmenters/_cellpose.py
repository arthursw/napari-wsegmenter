import numpy as np

model, last_parameters = None, None


def segment(image_path, segmentation_path, parameters):

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

    image = np.load(image_path)
    if model is None:
        return
    print("Computing segmentation...")
    masks, flows, styles, diams = model.eval(
        image, diameter=parameters["diameter"], channels=parameters["channels"]
    )
    print("segmentation finished.")
    np.save(segmentation_path, masks)
    return segmentation_path
