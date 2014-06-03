from __future__ import division as _division
from bpf4 import bpf
from peach import *
from collections import namedtuple as _namedtuple
import numpy as np
import warnings as _warnings
import platform as _platform
import logging as _logging

__all__ = "Spectrum fromtxt fromcsv fromsdif fromhdf5".split()

_logging.basicConfig(format = ">> %(message)s")
_logger = _logging.getLogger()
_logger.setLevel(_logging.WARN)  # leave this line uncommented to see only warnings

def _noop(*args, **kws):
    """
    This function is in place of some function which is not supported
    on your system, either because there are missing modules, or 
    because they are not supported in your platform
    """
    pass

Breakpoint = _namedtuple("Breakpoint", "freq amp phase bw")

###################################################################
#
#   Partial
#
###################################################################
class Partial(object):
    def __init__(self, id, times, freqs, amps, phase=None, bw=None):
        times = np.array(times, dtype=float)
        self.freq = bpf.core.Linear(times, freqs)
        self.amp = bpf.core.Linear(times, amps)
        self.phase = phase if phase else bpf.core.Const(0)
        self.bw = bw if bw else bpf.core.Const(0)
        self.id = id
        self.t0 = t0 = times[0]
        self.t1 = t1 = times[-1]
        self.duration = dur = t1 - t0 
        if dur <= 0:
            raise ValueError("A Partial should have a duration longer than 0")
        self._meanfreq = -1
        self._meanamp = -1
        self._wmeanfreq = -1
        
    @staticmethod
    def fromarray(data, partialid=0):
        """
        data is a 2D array with the shape (numbreakpoints, 5)
        columns: time freq amp phase bw
        """
        time  = data[:,0].copy()
        freq  = data[:,1].copy()
        amp   = data[:,2].copy()
        phase = data[:,3].copy()
        bw    = data[:,4].copy()
        return Partial(partialid, times, freqs, amps, phase, bw)
    
    def __repr__(self):
        return "Partial %d [%.4f:%.4f]" % (self.id, self.t0, self.t1)
    
    def __eq__(self, other):
        if self is other:
            return True
        t0, f0 = self.freq.points()
        _, a0 = self.amp.points()
        t1, f1 = other.freq.points()
        _, a1 = other.amp.points()
        return (t0 == t1).all() and (f0 == f1).all() and (a0 == a1).all()
    
    def data_at(self, t):
        return Breakpoint(self.freq(t), self.amp(t), self.phase(t), self.bw(t))
    
    @property
    def meanfreq(self):
        if self._meanfreq >= 0:
            return self._meanfreq
        mean = self.freq.integrate_between(self.t0, self.t1) / (self.t1 - self.t0)
        self._meanfreq = mean
        return mean
    
    @property
    def meanfreq_weighted(self):
        """
        weighted mean frequency
        """
        if self._wmeanfreq > 0:
            return self._wmeanfreq
        sumweight = self.amp.integrate_between(self.t0, self.t1)
        if sumweight > 0:
            out = (self.freq * self.amp).integrate_between(self.t0, self.t1)  / sumweight
        else:
            out = 0
        self._wmeanfreq = out
        return out
    
    @property
    def meanamp(self):
        if self._meanamp >= 0:
            return self._meanamp
        out = self.amp.integrate_between(self.t0, self.t1) / (self.t1 - self.t0)
        self._meanamp = out
        return out
    
    def __contains__(self, t):
        return self.t0 <= t <= self.t1

#############################################
# Filter presets
#############################################
FREQCURVES = {
    'instrumental': bpf.linear(0, -120, 3000, -120, 4500, -80, 6000, 0)
}

DURCURVES = {
    # x: duration, y: minimal average amplitude to not be filtered
    'low': bpf.linear(0, 0, 0.01, -20, 0.1, -40, 0.2, -80, 0.4, -120)
}

_SpectrumFilter = _namedtuple("_SpectrumFilter", "selected rejected")

#######################################################################
#
# Spectrum
#
#######################################################################
class Spectrum(object):
    def __init__(self, partials):
        """
        You normally dont create a Spectrum from scratch, but read it from a file
        (txt, sdif, or any sound file which can be analyzed)
        """
        self.partials = list(partials) if not isinstance(partials, list) else partials
        self.partials.sort(key=lambda p:p.t0)
        self.reset()
        
    def reset(self):
        self.t0 = min(p.t0 for p in partials)
        self.t1 = max(p.t1 for p in partials)
        
    def __repr__(self):
        return "Spectrum [%.4f:%.4f]: %d partials" % (self.t0, self.t1, len(self.partials))
    
    def __iter__(self):
        return iter(self.partials)
    
    def __eq__(self, other):
        for p0, p1 in zip(self, other):
            if p0 != p1:
                return False
        return True
    
    def partials_at(self, t):
        return self.partials_between(t, t + 1e-12)
    
    def partials_between(self, t0, t1):
        out = []; out_append = out.append
        if (t0+t1) * 0.5 < (self.t0 + self.t1) * 0.5:
            for partial in self.partials:
                if partial.t1 >= t0 and partial.t0 <= t1:
                    out_append(partial)
                if partial.t0 > t1:
                    break
        else:
            for partial in reversed(self.partials):
                if partial.t1 < t0:
                    break
                if partial.t0 <= t1 and partial.t1 >= t0:
                    out_append(partial)
            out.reverse()
        return out

    def chord_at(self, t, maxnotes=None, minamp=-50):
        """
        a quick way to query a spectrum at a given time
        """
        ps = self.partials_at(t)
        data = [(p.amp(t), p.freq(t)) for p in ps]
        data.sort(reverse=True)
        if maxnotes is not None:
            data = data[:maxnotes]
        minamp = db2amp(minamp)
        data2 = [(f2m(f), a) for a, f in data if a >= minamp]
        out = _Chord(data2)
        return out

    def filter_quick(self, mindur=0, minamp= -90, minfreq=0, maxfreq=24000):
        """
        intended for a quick filtering of undesired partials

        minamp (dB): 

        SEE ALSO: filter
        """
        no = []
        no_append = no.append
        for p in self.partials:
            if p.duration < mindur:
                no_append(p)
            elif p.meanamp() < minamp:
                no_append(p)
            elif not (minfreq < p.meanfreq() < maxfreq):
                no_append(p)
        return Spectrum(no)

    def filter(self, freqcurve='instrumental', durcurve='low'):
        """
        return too Spectrums, one which satisfies the given criteria, and the resisuum
        so that both reconstruct the original Spectrum
        
        freqcurve: a bpf or a string preset (see FREQCURVES)
        durcurve:  a bpf or a string preset (see DURCURVES)
        
        These curves determine how loud a partial must be for a given frequency or duration
        
        Example
        -------
        
        Filter out weak partials outside the range of musical instruments, 
        preparing for score transcription
        
        >>> partials = fromtxt("analysis.txt")
        >>> strong_partials, week_partials = partials.filter()

        """
        if isinstance(freqcurve, basestring):
            freqcurve = FREQCURVES.get(freqcurve)
            if not freqcurve:
                raise ValueError("the given freqcurve string is not an existing preset")
        if isinstance(durcurve, basestring):
            durcurve = DURCURVES.get(durcurve)
            if not durcurve:
                raise ValueError("the given duration curve is not an existing preset")
        above = []
        below = []
        freqcurve_amp = freqcurve.apply(db2amp).sampled(freqcurve.ntodx(200))
        if durcurve:
            durcurve_amp = durcurve.apply(db2amp).sampled(durcurve.ntodx(200))
        for p in self.partials:
            minamp = freqcurve_amp( p.wmeanfreq() )
            if durcurve:
                dur_minamp = durcurve_amp( p.duration )
                minamp = min(minamp, dur_minamp)
            if p.meanamp() < minamp:
                below.append( p )
            else:
                above.append( p )
        return _SpectrumFilter(self.__class__(above), self.__class__(below))
    
    def copy(self):
        return self.__class__(self.partials[:])
    
    def __getitem__(self, n):
        # TODO: support slicing
        return self.partials[n]
    
    def __len__(self):
        return len(self.partials)
    
    def show(self):
        self._show_in_spear()
        
    def _show_in_spear(self):
        outfile = 'tmp.txt'
        _write_partials_as_spear(self.partials, outfile)
        _call_spear_with(outfile)
        
    def write(self, outfile):
        """
        write the partial information as

        .txt  : in the format defined by spear
        .sdif : SDIF format (not implemented) #TODO
        .csv  : dump all breakpoints to csv with columns [partial-id, time, freq, amp]
        .hdf5 : 
        """
        base, ext = os.path.splitext(outfile)
        if ext == '.txt':
            _write_partials_as_spear(self.partials, outfile)
        elif ext == '.hdf5' or ext == '.h5':
            _write_as_hdf5(self, outfile)
        elif ext == ".csv":
            _write_as_csv(self, outfile)
        else:
            raise RuntimeError("Format not supported")
    # <--------------------------------- END Spectrum

#################################################
# HELPERS
#################################################
def _call_spear_with(path):
    def osx(path):
        import subprocess
        subprocess.call('open -a SPEAR "%s"' % path, shell=True)
    def win(path):
        raise NotImplemented
    f = {
        'Darwin' : osx,
        'Windows' : win
    }.get(_platform.system())
    if f:
        f(path)

################################################
# IO
################################################
def _write_partials_as_spear(spectrum, outfile):
    f = open(outfile, 'wb')
    f_write = f.write
    f_write("par-text-partials-format\n")
    f_write("point-type time frequency amplitude\n")
    f_write("partials-count %d\n" % len(spectrum))
    f_write("partials-data\n")
    column_stack = np.column_stack
    for p in spectrum:
        times, freqs = p.freq.points()
        _, amps = p.amp.points()
        data = column_stack((times, freqs, amps)).flatten()
        f_write("%d %d %f %f\n" % (p.id, len(times), p.t0, p.t1))
        data_as_string = (str(n) for n in data)
        datastr = " ".join(str(n) for n in data)
        f_write(datastr + '\n')

def _write_as_hdf5(spectrum, outfile):
    try:
        import pandas
        from pandas import DataFrame
    except ImportError:
        raise ImportError("pandas not found. hdf5 is not supported!")
    store = pandas.HDFStore(outfile)
    def partial_id(p):
        return "p%d" % (int(p.id))
    def as_dataframe(p):
        time, freq = p.freq.points()
        _, amp = p.amp.points()
        return DataFrame({'time':time, 'amp':amp, 'freq':freq})
    for p in spectrum:
        store.put(partial_id(p), as_dataframe(p))
    store.flush()
    store.close()

def _write_as_csv(spectrum, outfile):
    from cStringIO import StringIO
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

def fromcsv(path):
    f = open(path)
    header = f.readline()
    columns = header.strip().split(",")
    assert columns[:4] == ["id", "time", "freq", "amp"]
    current_id = -1
    partials = []
    for line in f:
        uid, time, freq, amp = line.strip().split(",")[:4]
        uid = int(uid)
        if uid != current_id:
            if current_id >= 0:
                assert len(times) > 0
                partial = Partial(current_id, times, freqs, amps)
                current_id = uid
                partials.append(partial)
                times, freqs, amps = [], [], []
            else:
                times, freqs, amps = [], [], []
                current_id = uid
        else:
            times.append(time)
            freqs.append(freq)
            amps.append(amp)       
    partials.append( Partial(current_id, times, freqs, amps) )
    assert partials
    return Spectrum(partials)

def fromtxt(path, debug=True):
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
            for i in range(10):
                same = times[1:] - times[:-1] == 0
                if same.any():
                    _logger.warn("duplicate points found")
                    times[1:][same] += EPSILON
                else:
                    break
        dur = times[-1] - times[0]
        if dur > 0:
            partial = Partial(partial_id, times, freqs, amps)
            partials.append(partial)
        else:
            skipped += 1
        npartials -= 1
    if skipped:
        _logger.warn("Skipped %d partials without duration" % skipped)
    return Spectrum(partials)

def fromhdf5(path, method="fast"):
    try:
        import pandas
    except ImportError:
        raise ImportError("Could not find pandas. HDF5 support is not available!")
    store = pandas.HDFStore(path, mode="r")
    partials = []
    partials_append = partials.append
    if method == "fast":
        for key, value in store.iteritems():
            a = value.block0_values.read()
            amp  = a[:,0]
            freq = a[:,1]
            time = a[:,2]
            partial_id = int(key[2:])  # get rid of the /p
            partial = Partial(partial_id, time, freq, amp)
            partials_append(partial)
    else:
        for key in store.keys():
            partial_id = int(key[2:])  # get rid of the /p
            frame = store[key]
            partial = Partial(partial_id, frame.time, frame.freq, frame.amp)
            partials_append(partial)
    store.close()
    return Spectrum(partials)
    
def fromsdif(path):
    return read_sdif(path)
    
def _partial2dataframe(partial):
    time, freq = partial.freq.points()
    _, amp = partial.amp.points()
    return DataFrame({'time':time, 'freq':freq, 'amp':amp})
        
try:
    import pandas
    from pandas import DataFrame
    partial2dataframe = _partial2dataframe
except ImportError:
    partial2dataframe = _noop

