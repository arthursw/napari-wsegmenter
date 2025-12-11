try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import CellposeWidget, SamWidget, StardistWidget
from ._widget_simple import cellpose, exit_button, sam, stardist

__all__ = (
    "stardist",
    "cellpose",
    "sam",
    "exit_button",
    "CellposeWidget",
    "StardistWidget",
    "SamWidget",
)
