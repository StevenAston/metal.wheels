"""metal.wheels — long-horizon metal stock-flow and maintenance-energy model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("metalwheels")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
