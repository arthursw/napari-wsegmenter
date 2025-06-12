# from napari_wsegmenter._widget_shared_memory import (
#     segment_widget_shared_memory,
# )
# @pytest.mark.parametrize(
#     "shape, model_name",
#     [
#         ((100, 100), "stardist"),
#         ((64, 64), "cellpose"),
#         ((128, 128), "sam"),
#     ],
# )
# def test_segment_widget_shared_memory(shape, model_name):
#     im_data = np.random.random(shape)
#     widget = segment_widget_shared_memory()
#     thresholded = widget(im_data, model_name)
#     assert thresholded.shape == im_data.shape
import logging

import numpy as np
import pytest

from napari_wsegmenter._widget_qt_shared_memory import (
    ImageSegmenter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "shape, model_name",
    [
        ((100, 100), "StarDist"),
        # ((64, 64), "Cellpose"),
        # ((128, 128), "SAM"),
    ],
)
def test_segment_widget_qt_shared_memory(
    make_napari_viewer, shape, model_name
):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random(shape))

    # create our widget, passing in the viewer
    widget = ImageSegmenter(viewer)
    widget._segmenter_combo.setCurrentText(model_name)
    widget._update_image_layers()
    widget._image_layer_combo.setCurrentIndex(0)

    # call our widget method
    widget.segment()

    segmented_layers = [
        layer
        for layer in viewer.layers
        if layer.__class__.__name__ == "Labels"
        and layer.name.endswith("_segmented")
    ]

    assert segmented_layers[0].data.shape == shape

    widget._release_shared_memory()
