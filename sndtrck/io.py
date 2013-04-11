import os
import backend_loris
if backend_loris.is_available():
    read_sdif = backend_loris.read_sdif
del backend_loris
from collections import namedtuple

Sample = namedtuple("Sample", "samples sr")

def readsnd(path):
    """
    read the soundfile and return a tuple (samples, sr) as float64
    """
    try:
        import sndfileio
        return sndfileio.read_sndfile(path)
    except ImportError:
        pass
    try:
        from scikits import audiolab
        snd = audiolab.Sndfile(path)
        data = snd.read_frames(snd.nframes)
        sr = snd.samplerate
        return Sample(data, sr)
    except ImportError:
        pass
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        try:
            # the normal wave module is too buggy when reading 24bits
            from scipy.io import wavfile 
            import wave
            bits = wave.open(path).getsampwidth() * 8 # 1:8bit . 2:16bit . 3:24bit
            sr, data = wavfile.read(path)
            if bits == 8:
                data = data / float(2**7)
            elif bits == 16:
                data = data / float(2**15)
            elif bits == 24:
                data = data / float(2**31)  # it is read into an int32
            elif bits == 32:
                raise ValueError("scipy.io.wavfile does not handle 32 bit wav-files properly.")
            return Sample(data, sr)
        except ImportError:
            pass
    raise IOError("could not read the file %s. Install sndfileio or scikits.audiolab and try again" % path)





