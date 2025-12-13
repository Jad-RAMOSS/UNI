"""Audio endpoint helpers built on top of pycaw."""

from ctypes import POINTER, cast

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

__all__ = ["get_audio_endpoint"]


def get_audio_endpoint():
    """Return the default speaker endpoint for volume control."""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))
