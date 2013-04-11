from . import bpfpartials
    
def is_available():
    try:
        import loristrck
        return True
    except ImportError:
        return False

def _readsnd(snd):
    if isinstance(snd, tuple):
        return snd
    from e import sndfileio
    return sndfileio.read_sndfile(snd)

def analyze(snd, resolution, window_width=None, **config):
    """
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
    samples, sr = _readsnd(snd)
    if window_width is None:
        window_width = resolution * 2 # original Loris behaviour
    partials = loristrck.analyze(samples, sr, resolution, window_width, **config)
    return _partialsgen_to_spectrum(partials)

def _partialsgen_to_spectrum(partials):
    """
    partials: as returned by loristrck.analyze, a generator of label, data
    """
    partial_list = []
    fromarray = bpfpartials.Partial.fromarray
    for label, data in partials:
        partial = fromarray(data)
        partial_list.append(partial)
    return bpfpartials.Spectrum(partial_list)

def read_sdif(sdiffile):
    import loristrckr
    partials = loristrckr.read_sdif(sdiffile)
    return _partialsgen_to_spectrum(partials)




