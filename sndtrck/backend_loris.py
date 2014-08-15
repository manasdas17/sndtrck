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
"""
from . import bpfpartials
from . import io
    
def is_available():
    try:
        import loristrck
        return True
    except ImportError:
        return False

def analyze(snd, resolution, window_width=None, verbose=False, **config):
    """
    Analyze a soundfile for partial tracking. Depends on loristrck

    snd {string or (numpy.array, samplerate)} --> the soundfile as path or the samples and samplerate
    resolution   {Hz} --> the resolution of the analysis
    window_width {Hz} --> the width of the analysis window. If not given, it is 2*resolution
    
    Other keywords will be passed directly to the Loris analyzer. Possible keywords are:
      
    freq_drift --> max. drift in frequency between 2 breakpoints. Used in the tracking-phase
    hop_time   --> hop time in seconds
    freq_drift --> frequency drift in Hz is the maximum difference in frequency between 
                   consecutive Breakpoints
    sidelobe_level --> positive dB. Sets the shape of the Kaiser window used
    amp_floor  --> only breakpoint above this amplitude are kept
    """
    import loristrck
    if verbose:
        print("reading sndfile")
    samples, sr = io.sndread(snd)
    if window_width is None:
        window_width = resolution * 2 # original Loris behaviour
    if verbose: print("analyzong samples")
    partials = loristrck.analyze(samples, sr, resolution, window_width, **config)
    if verbose: print("converting to Spectrum")
    return _partialsgen_to_spectrum(partials)

def _partialsgen_to_spectrum(partials, verbose=True):
    """
    partials: as returned by loristrck.analyze, a generator of label, data
    """
    partial_list = []
    fromarray = bpfpartials.Partial.fromarray
    for label, data in partials:
        times = data[:,0]
        if len(times) > 1 and times[-1] - times[0] > 0:
            partial = fromarray(data)
            partial_list.append(partial)
        else:
            if verbose:
                print("skipping short partial")
    return bpfpartials.Spectrum(partial_list)

def read_sdif(sdiffile):
    import loristrckr
    partials = loristrckr.read_sdif(sdiffile)
    return _partialsgen_to_spectrum(partials)

