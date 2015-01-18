"""
loris backend based on loristrck

Will only work if loristrck is available. Loristrck is a simple wrapper 
around the partial tracking library Loris, which differs from its built-in 
python bindings in that it is not swig generated but implemented in cython. 
It does not need Loris itself to be installed: it links directly to it and
is compiled on the package itself. This makes installation much easier and
reliable, since there are no PATH problems at compile or at runtime.
Loris is only used to analyze the sound and is converted to an agnostic 
data representation based on numpy arrays. This makes it easier to manipulate
(the Loris bindings are not very nice to use from a python stand-point)

A backend should implement:

is_available() -> bool
get_info() --> dict 
               {
                'analyze': bool,
                'read_sdif': bool,
                'write_sdif': bool
               }

"""
import numpy as np
import sndfileio
import warnings as _warnings

from .errors import *
from . import config as _config
from . import lib


try:
    import loristrck
    AVAILABLE = True
except ImportError:
    _warnings.warn("loristrck is not available, the loris backend cannot be used")
    AVAILABLE = False


# A backend should implement these functionality:
#    * is_available() -> bool
#    * get_info()     -> dict with keys defining the available functionality

    
def is_available():
    return AVAILABLE


def get_info():
    return {
        'analyze': True,
        'read_sdif': True,
        'write_sdif': True,
        'analysis_options': {
            'freq_drift': 'max. drift in frequency between 2 breakpoints',
            'hop_time': 'hop time in seconds',
            'sidelobe_level': 'sets the dhape of the Kaiser window used (in positive dB)',
            'amp_floor': 'only breakpoints above this amplitude are kept'
        }
    }


def analyze(snd, resolution, window_width=None, verbose=False, **kws):
    """
    Analyze a soundfile for partial tracking. Returns partialdata to be passed
    to fromarray

    snd {string or (numpy.array, samplerate)} --> the soundfile as path or the samples and samplerate
    resolution   {Hz} --> the resolution of the analysis
    window_width {Hz} --> the width of the analysis window. If not given, it is 2*resolution
    
    Other keywords will be passed directly to the Loris analyzer. Possible keywords are:
      
    freq_drift :: max. drift in frequency between 2 breakpoints. Used in the tracking-phase
    hop_time   :: hop time in seconds
    freq_drift :: frequency drift in Hz is the maximum difference in frequency between 
                   consecutive Breakpoints
    sidelobe_level :: positive dB. Sets the shape of the Kaiser window used
    amp_floor  :: only breakpoint above this amplitude are kept

    monochannel:: The channel (as int) to use if snd is not mono, or 'mix'
                  to mix down all channels
    """
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")

    def warn_multichannel():
        _warnings.warn(
            "Soundfile has {numchannels} channels. "
            "it will be converted to mono (channel: {monochannel})".format(**locals())
        )

    monochannel = kws.pop('monochannel', _config.CONFIG['monochannel'])
    samples, sr = lib.sndreadmono(snd, channel=monochannel)
    samples = lib.as_c_contiguous(samples)
        
    # original Loris behaviour
    if window_width is None:
        window_width = resolution * 2 
    partialdata = loristrck.analyze(samples, sr, resolution, window_width, **kws)
    return partialdata


def read_sdif(sdiffile):
    """
    reads sdiffile, returns a list of (matrices, label) as expected by fromarray

    To construct a Spectrum, call fromarray(read_sdif(sdiffile))
    """
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")
    data = loristrck.read_sdif(sdiffile)
    return data


def write_sdif(outfile, matrices, labels=None, rbep=True, fadetime=0):
    """
    partials: a seq. of (label, matrix) where matrix is a 2D array with columns 
              [time, freq, amp, phase, bw]

    """
    if not AVAILABLE:
        raise BackendNotAvailable("loristrck not available")
    loristrck.write_sdif(outfile, matrices, labels=labels, rbep=rbep, fadetime=fadetime)
