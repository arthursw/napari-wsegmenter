# napari-wsegmenter worker

This internal distribution contains the qualified worker targets declared by the napari-wsegmenter manifest.
Napari installs it into each managed plugin environment; it is not a host-side plugin API or a separately published package.

The worker targets accept NumPy arrays and plain dictionaries, lazily import the segmentation framework supplied by their environment, and return NumPy label arrays.
They use the injected `napari_context` only for progress and cooperative cancellation.
