from . import backend_loris
from .errors import *
from collections import namedtuple as _namedtuple
from . import log as _log
from . import lib as _lib
import numpy as np

"""
IO

This is a 'floor' module, in-out is done via built-in datatypes

Conversions to Spectrum are done in spectrum.py

NB: we implement out conversions here because we don't need to
import spectrum for that, we convert a Spectrum to an array
and go with that. For that reason the to* functions can be
exported as is, the from* functions need to be wrapped one
level higher

For soundfile io, use sndfileio directly
"""

_logger = _log.get_logger()

_Partial = _namedtuple("_Partial", "label time freq amp phase bw")


def analyze(snd, resolution, window_width=None, verbose=False, **config):
    """
    returns the analyzed spectrum as needed by fromarray. See
    the backends for configuration options
    """
    for backend in [backend_loris]:
        if backend.is_available():
            partialdata = backend.analyze(
                snd, resolution, window_width=window_width,
                verbose=verbose, **config)
            return partialdata
    raise FunctionalityNotAvailable("no backend implements analysis!")


@_lib.public
def analysis_options(display=True):
    """
    Returns a dictionary of options available for the used backend
    to be passed as keyword arguments to the 'analyze' function

    Returns None if no backend supports analysis
    """
    for backend in [backend_loris]:
        if backend.is_available() and backend.get_info().get('analyze'):
            options = backend.get_info().get('analysis_options', {})
            if options and display:
                for option, doc in options.iteritems():
                    print("{option}: {doc}".format(option=option.ljust(16), doc=doc))
            return options


@_lib.public
def tosdif(spectrum, outfile, rbep=True, fadetime=0):
    """
    Write this spectrum as SDIF

    rbep: if True, use the given timing, do not resample to a common-grid
    fadetime: if > 0, add a fadein-out to the partials when the first or
              last breakpoint has an amplitude > 0
    """
    outfile = _lib.normalizepath(outfile)
    matrices, labels = spectrum.toarray()
    for backend in [backend_loris]:
        if backend.is_available() and backend.get_info()['write_sdif']:
            return backend.write_sdif(
                outfile, matrices, labels=labels, rbep=rbep, fadetime=fadetime)
    raise FunctionalityNotAvailable("no backend implements writing to sdif")


def fromsdif(sdiffile):
    """
    Reads a SDIF file (1TRC or RBEP)

    Returns a list of (matrices, label) as expected by fromarray

    Raises FunctionalityNotAvailable if no backend implements this
    """
    from . import backend_loris
    backends = [backend_loris]
    for backend in backends:
        if backend.is_available() and backend.get_info()['read_sdif']:
            return backend.read_sdif(sdiffile)
    raise FunctionalityNotAvailable("No Backend implements SDIF reading")


def _newpartial(label, times, freqs, amps, phase=None, bw=None):
    return _Partial(label, time, freq, amp, phase, bw)


def fromtxt(path):
    """
    Read a Spectrum from a txt file in the format used by SPEAR

    returns a generator of _Partials
    """
    f = open(path)
    it = iter(f)
    it.next()
    it.next()
    npartials = int(it.next().split()[1])
    it.next()
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
        # if this is the case, shift the second point to the right 
        # a minimal amount
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
            #partial = _spectrum.Partial(partial_id, times, freqs, amps)
            partial = _newpartial(partial_id, times, freqs, amps)
            yield partial
        else:
            skipped += 1
        npartials -= 1
    if skipped:
        _logger.warn("Skipped %d partials without duration" % skipped)


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
            partial = _newpartial(partial_id, time, freq, amp)
            yield partial
    store.close()


@_lib.public
def tospear(spectrum, outfile, use_comma=False):
    """
    writes the partials in the text format defined by SPEAR 
    (Export Format/Text - Partials)

    The IDs of the partials are lost, partials are enumerated
    in the order defined in the Spectrum
    """
    outfile = _lib.normalizepath(outfile)
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


@_lib.public
def tohdf5(spectrum, outfile):
    try:
        import pandas
    except ImportError:
        raise ImportError("pandas not found. hdf5 is not supported!")
    outfile = _lib.normalizepath(outfile)
    store = pandas.HDFStore(outfile)

    def as_dataframe(p):
        time, freq = p.freq.points()
        _, amp = p.amp.points()
        return pandas.DataFrame({b'time': time, b'amp': amp, b'freq': freq})

    for i, p in enumerate(spectrum):
        store.put("i_%d" % i, as_dataframe(p))
    labels = pandas.DataFrame(
        {'labels': np.array([p.id for p in spectrum], dtype=int)}
    )
    store.put(b'labels', labels)
    store.flush()
    store.close()

# ------------------------------------------------------


def readspectrum(path):
    """
    Read a spectrum, returns a gen of _Partials

    Supported formats: txt (spear), sdif, csv, hdf5
    """
    func = {
        '.txt': fromtxt,
        '.hdf5': fromhdf5,
        '.h5': fromhdf5,
        '.sdif': fromsdif,
    }
    ext = os.path.splitext(path)[1]
    if ext in func:
        return func[ext](path)
    else:
        raise ValueError("Format not supported")

