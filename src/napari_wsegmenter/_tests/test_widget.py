import logging

import numpy as np

from napari_wsegmenter._widget_simple import cellpose, exit_environments

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_cellpose(make_napari_viewer):
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100)))

    my_widget = cellpose()

    segmentation, _, _ = my_widget(viewer.layers[0].data)
    assert segmentation.shape == layer.data.shape[:2]

    exit_environments()
