"""Audio endpoint helpers built on top of pycaw."""

from pycaw.pycaw import AudioUtilities

__all__ = ["get_audio_endpoint"]


def get_audio_endpoint():
    """
    Return the default speaker endpoint for volume control.
    Compatible with older pycaw versions.
    """
    device = AudioUtilities.GetSpeakers()

    # OLD pycaw exposes volume like this
    volume = device.EndpointVolume
    return volume
