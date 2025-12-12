from napari import Viewer, run

viewer = Viewer()
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Cellpose segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Cellpose segmenter pickle"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Cellpose segmenter local"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Stardist segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "SAM segmenter"
)
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-wsegmenter", "Exit"
)

# dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
#     "napari-wsegmenter", "Sam qwidget"
# )
# dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
#     "napari-wsegmenter", "Cellpose qwidget"
# )
# dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
#     "napari-wsegmenter", "Stardist qwidget"
# )

run()
