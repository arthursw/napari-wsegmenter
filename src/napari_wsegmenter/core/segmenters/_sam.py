def _create_segmentation(annotations):
    import numpy as np

    image = np.ones(
        (
            annotations[0]["segmentation"].shape[0],
            annotations[0]["segmentation"].shape[1],
        )
    )
    for i, annotation in enumerate(annotations):
        image[annotation["segmentation"]] = i + 1
    return image


predictor, mask_generator, last_parameters = None, None, None


def segment(image, parameters):
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_hf

    global predictor, mask_generator, last_parameters

    device = "cuda" if parameters["use_gpu"] else "cpu"
    if (
        predictor is None
        or last_parameters is None
        or last_parameters["use_gpu"] != parameters["use_gpu"]
    ):
        predictor = build_sam2_hf(
            "facebook/sam2.1-hiera-large",
            device=torch.device(device),
            apply_postprocessing=False,
        )

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        if mask_generator is None or last_parameters != parameters:
            mask_generator = SAM2AutomaticMaskGenerator(
                predictor,
                points_per_side=parameters["points_per_side"],
                pred_iou_thresh=parameters["pred_iou_thresh"],
                stability_score_thresh=parameters["stability_score_thresh"],
            )
        last_parameters = parameters
        masks = mask_generator.generate(image)
        return _create_segmentation(masks)
