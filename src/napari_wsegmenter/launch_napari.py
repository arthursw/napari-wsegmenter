import logging

from napari import Viewer, run
from wetlands import logger

logger.setLogLevel(logging.DEBUG)

viewer = Viewer()
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Segmenter"
)
run()
