"""
io.py 
"""
from .errors import *
from . import spectrum as _spectrum
from . import log
import numpy as np
import os

_logger = log.get_logger()

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

def fromsdif(sdiffile):
    """
    Reads a SDIF file (1TRC or RBEP)

    Returns 

    Returns: A Spectrum instance

    Raises FunctionalityNotAvailable if no backend implements this
    """
    from . import backend_loris
    backends = [backend_loris]
    for backend in backends:
        if backend.is_available() and backend.get_info()['read_sdif']:
            return backend.read_sdif(sdiffile)
    raise FunctionalityNotAvailable("No Backend implements SDIF reading")

def tosdif(spectrum, outfile, rbep=True, fadetime=0):
    """
    Write this spectrum as SDIF

    rbep: if True, use the given timing, do not resample to a common-grid
    fadetime: if > 0, add a fadein-out to the partials when the first or 
              last breakpoint has an amplitude > 0
    """
    from .import backend_loris
    for backend in [backend_loris]:
        if backend.is_available() and backend.get_info()['write_sdif']:
            matrices, labels = spectrum.toarray()
            return backend.write_sdif(outfile, matrices, labels=labels, rbep=rbep, fadetime=fadetime)
    raise FunctionalityNotAvailable("no backend implements writing to sdif")

def fromtxt(path):
    """
    Read a Spectrum from a txt file in the format used by SPEAR
    """
    f = open(path)
    it = iter(f)
    it.next(); it.next()
    npartials = int(it.next().split()[1])
    it.next()
    partials = []
    EPSILON = 1e-10
    nextline = it.next
    skipped = 0
    while npartials > 0:
        partial_id = float(nextline().split()[0])
        data = np.fromstring(nextline(), sep=" ", dtype=float)
        times = data[::3]
        freqs = data[1::3]
        amps = data[2::3]
        # check if any point has the same time as the previous one
        # if this is the case, shift the second point to the right a minimal amount
        if len(times) > 2:
            for _ in range(10):
                same = times[1:] - times[:-1] == 0
                if same.any():
                    _logger.warn("duplicate points found")
                    times[1:][same] += EPSILON
                else:
                    break
        dur = times[-1] - times[0]
        if dur > 0:
            partial = _spectrum.Partial(partial_id, times, freqs, amps)
            partials.append(partial)
        else:
            skipped += 1
        npartials -= 1
    if skipped:
        _logger.warn("Skipped %d partials without duration" % skipped)
    return _spectrum.Spectrum(partials)


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

def fromcsv(path):
    """
    Read a Spectrum from a csv file

    path: the path to a .csv file with columns ID, TIME, FREQ, AMP, ...
          (other columns will not be taken into account)
          Each row contains a breakpoint, the ID identifies the partial to which
          it belongs

    Returns: a Spectrum
    """
    f = open(path)
    header = f.readline()
    columns = header.strip().split(",")
    assert columns[:4] == ["id", "time", "freq", "amp"]
    partials = {}
    for line in f:
        uid, time, freq, amp = line.strip().split(",")[:4]
        uid = int(uid)
        partial = partials.get(uid, [])
        partial.append((time, freq, amp))
    labels, matrices = partials.keys(), partials.values()
    matrices = [np.array(matrix, dtype=float) for matrix in matrices]
    return _spectrum.fromarray(zip(matrices, labels))

def tocsv(spectrum, outfile):
    stream = open(outfile, "w")
    column_stack = np.column_stack
    savetxt = np.savetxt
    def writepartial(stream, partial):
        time, freq = partial.freq.points()
        _, amp = partial.amp.points()
        data = column_stack((np.ones_like(amp, dtype=int)*partial.id, time, freq, amp))
        savetxt(stream, data, delimiter=",", fmt=["%d", "%.18e", "%.18e", "%.18e"])
    stream.write("id,time,freq,amp\n")
    for p in spectrum:
        writepartial(stream, p)
    stream.flush()
    stream.close()

def tospear(spectrum, outfile, use_comma=True):
    """
    writes the partials in the text format defined by SPEAR 
    (Export Format/Text - Partials)

    The IDs of the partials are lost, partials are enumerated
    in the order defined in the Spectrum
    """
    f = open(outfile, 'wb')
    f_write = f.write
    f_write("par-text-partials-format\n")
    f_write("point-type time frequency amplitude\n")
    f_write("partials-count %d\n" % len(spectrum))
    f_write("partials-data\n")
    column_stack = np.column_stack
    for i, p in enumerate(spectrum):
        times, freqs = p.freq.points()
        _, amps = p.amp.points()
        data = column_stack((times, freqs, amps)).flatten()
        assert len(data) == 3 * p.numbreakpoints
        header = "%d %d %f %f\n" % (i, len(times), p.t0, p.t1)
        datastr = " ".join("%f" % n for n in data)
        if use_comma:
            header = header.replace(".", ",")
            datastr = datastr.replace(".", ",")
        f_write(header)
        f_write(datastr)
        f_write('\n')

def tohdf5(spectrum, outfile):
    try:
        import pandas
        from pandas import DataFrame
    except ImportError:
        raise ImportError("pandas not found. hdf5 is not supported!")
    store = pandas.HDFStore(outfile)
    def as_dataframe(p):
        time, freq = p.freq.points()
        _, amp = p.amp.points()
        return DataFrame({b'time':time, b'amp':amp, b'freq':freq})
    for i, p in enumerate(spectrum):
        store.put("i_%d" % i, as_dataframe(p))
    labels = pandas.DataFrame({'labels': np.array([p.id for p in spectrum], dtype=int)})
    store.put(b'labels', labels)
    store.flush()
    store.close()

def fromhdf5(path):
    try:
        import pandas
    except ImportError:
        raise ImportError("Could not find pandas. HDF5 support is not available!")
    store = pandas.HDFStore(path, mode="r")
    partials = []
    partials_append = partials.append
    Partial = _spectrum.Partial
    labels_data = store.get('labels', None)
    if labels_data:
        labels = labels_data[:, 0]
    else:
        labels = None
    for key, value in store.iteritems():
        if key.startswith("i_"):
            a = value.block0_values.read()
            amp  = a[:,0]
            freq = a[:,1]
            time = a[:,2]
            partial_index = int(key[2:])  # get rid of the /p
            partial_id = labels[partial_index] if labels else 0
            partial = Partial(partial_id, time, freq, amp)
            partials_append(partial)
    store.close()
    return _spectrum.Spectrum(partials)


def readspectrum(path):
    """
    Read a spectrum 

    Supported formats: txt (spear), sdif, csv, hdf5
    """
    func = {
        '.txt' : fromtxt,
        '.hdf5': fromhdf5,
        '.h5'  : fromhdf5,
        '.sdif': fromsdif,
        '.csv' : fromcsv
    }
    ext = os.path.splitext(path)[1]
    if ext in func:
        return func[ext](path)
    else:
        raise ValueError("Format not supported")
        
    