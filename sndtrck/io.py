from . import backend_loris
if backend_loris.is_available():
    sdifread = backend_loris.read_sdif
del backend_loris

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

def _check_sndfileio():
    "check if sndfileio is present"
    try:
        import sndfileio
    except ImportError:
        _sndfileio_msg()
        raise ImportError("sndfileio is needed to read or write sound-files")

def sndread(path):
    """
    read the soundfile and return a tuple (samples, sr) as float64
    """
    _check_sndfileio()
    return sndfileio.sndread(path)
    
def sndwrite(samples, samplerate, path):
    """
    write the samples as a soundfile. Format and encoding are 
    determined by the extension of the outfile and by the 
    bitdepth of the data.
    """
    _check_sndfileio()
    return sndfileio.sndwrite(samples, samplerate, path)
