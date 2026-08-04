try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import CellposeWidget, SamWidget, StardistWidget, ThresholdWidget

__all__ = (
    "CellposeWidget",
    "StardistWidget",
    "SamWidget",
    "ThresholdWidget",
)
