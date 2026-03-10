from importlib.metadata import PackageNotFoundError, version

from .profile import profile_macs

__all__ = ["profile_macs"]

try:
    __version__ = version("torchprofile")
except PackageNotFoundError:
    __version__ = "0.0.0"
