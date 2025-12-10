import logging

from napari import Viewer, run

logging.getLogger("wetlands").addHandler(logging.StreamHandler())
logging.getLogger("wetlands").setLevel(logging.DEBUG)

viewer = Viewer()
# dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget("napari-wsegmenter")
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "SAM segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Cellpose segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Stardist segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Exit"
)
run()
