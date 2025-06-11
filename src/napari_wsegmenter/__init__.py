try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._widget import (
    segment_widget,
)
from ._widget_shared_memory import (
    segment_widget_shared_memory,
)

__all__ = (
    "segment_widget",
    "segment_widget_shared_memory",
)
