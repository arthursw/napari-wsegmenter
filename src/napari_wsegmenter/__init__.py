try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget_qt import (
    SegmenterWidget,
)

__all__ = ("SegmenterWidget",)
