import sys
from pathlib import Path

import numpy as np


def _segment(module_name, image, parameters) -> np.ndarray | None:
    import importlib

    module = importlib.import_module(module_name)
    return module.segment(image, parameters)


def segment_files(module_name, image_path, segmentation_path, parameters):
    image = np.load(image_path)
    segmentation = _segment(module_name, image, parameters)
    if segmentation is not None:
        np.save(segmentation_path, segmentation)
        return segmentation_path


def segment_shared_memory(
    module_name, shared_image, shared_segmentation, parameters
):

    sys.path.append(str(Path(__file__).parent.parent))
    from _memory_manager import get_shared_array  # type: ignore

    with get_shared_array(shared_image) as image:
        labels = _segment(module_name, image, parameters)
        with get_shared_array(shared_segmentation) as segmentation:
            segmentation[:] = labels
    return shared_segmentation
