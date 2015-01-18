import sys as sys
import os as os
import sndfileio as _sndfileio
from . import config


def public(f):
    """
    Use a decorator to avoid retyping function/class names.

    Keeps __all__ updated

    * Based on an idea by Duncan Booth:
      http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    * Improved via a suggestion by Dave Angel:
      http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
    """
    publicapi = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ not in publicapi:  # Prevent duplicates if run from an IDE.
        publicapi.append(f.__name__)
    return f


def normalizepath(path):
    return os.path.abspath(os.path.expanduser(path))


def sndreadmono(path, channel=None):
    """
    Read the soundfile and return a tuple (samples, sr) as float64
    If soundfile is not mono, convert it to mono.

    channel: can be a channel number or 'mix' to mix-down all channels
             None to use the default defined in config
    """
    samples, sr = _sndfileio.sndread(path)
    if channel is None:
        channel = config.CONFIG['monochannel']
    monosamples = _sndfileio.asmono(samples, channel)
    return monosamples, sr


def as_c_contiguous(a):
    """
    make sure that array is C-contiguous
    """
    if not a.flags.c_contiguous:
        return a.copy()
    return a
