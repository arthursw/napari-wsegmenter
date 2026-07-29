from napari import Viewer, run

viewer = Viewer()
for widget_name in ("Cellpose", "StarDist", "SAM"):
    viewer.window.add_plugin_dock_widget(
        "napari-wsegmenter",
        widget_name,
    )

run()
