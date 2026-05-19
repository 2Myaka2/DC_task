"""NaPi2b residue interaction network clustering pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napi2b-vkr")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
