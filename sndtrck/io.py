from .errors import *
"""
io.py 

This module should not depend on any internal module other than .errors 
"""

def _sndfileio_msg():
    print """
We depend on sndfileio for reading and writing soundfiles. 

Install it with pip:

$ pip install sndfileio

or git:

$ git clone https://github.com/gesellkammer/sndfileio.git
$ cd sndfileio
$ python setup.py install
"""

try:
    import sndfileio
except ImportError:
    _sndfileio_msg()
    raise ImportError("sndfileio could not be found")

def sdifread(sdiffile):
    """
    Reads a SDIF file.

    Returns: A Spectrum instance

    Raises FunctionalityNotAvailable if no backend implements this
    """
    from . import backend_loris
    if backend_loris.is_available():
        return backend_loris.read_sdif(sdiffile)
    else:
        raise FunctionalityNotAvailable("No Backend implements SDIF reading")

def sndread(path):
    """
    read the soundfile and return a tuple (samples, sr) as float64
    """
    return sndfileio.sndread(path)
    
def sndwrite(samples, samplerate, path):
    """
    write the samples as a soundfile. Format and encoding are 
    determined by the extension of the outfile and by the 
    bitdepth of the data.
    """
    return sndfileio.sndwrite(samples, samplerate, path)
