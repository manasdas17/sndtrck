from __future__ import division as _division
from bpf4 import bpf
from peach import *
from collections import namedtuple as _namedtuple
import numpy as np
import warnings as _warnings
import platform as _platform

from . import music as _music
from .log import get_logger

_logger = get_logger()


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
        def linear(a):
            return bpf.core.Linear(times, a)
        self.freq = linear(freqs)
        self.amp = linear(amps)
        self.phase = linear(phase) if phase is not None else bpf.core.Const(0)
        self.bw = linear(bw) if bw is not None else bpf.core.Const(0)
        self.id = id
        self.t0 = t0 = times[0]
        self.t1 = t1 = times[-1]
        self.duration = dur = t1 - t0 
        if dur <= 0:
            raise ValueError("A Partial should have a duration longer than 0")
        self._meanfreq = -1
        self._meanamp = -1
        self._wmeanfreq = -1

    @property
    def numbreakpoints(self):
        return len(self.freq.points()[0])
        
    @staticmethod
    def fromarray(data, partialid=0):
        """
        data is a 2D array with the shape (numbreakpoints, 5)
        columns: time freq amp phase bw
        """
        def as_c_contiguous(a):
            if not a.flags.c_contiguous:
                return a.copy()
            return a
        time  = as_c_contiguous(data[:,0])
        freq  = as_c_contiguous(data[:,1])
        amp   = as_c_contiguous(data[:,2])
        phase = as_c_contiguous(data[:,3])
        bw    = as_c_contiguous(data[:,4])
        return Partial(partialid, time, freq, amp, phase, bw)

    def toarray(self):
        time, freq = self.freq.points()
        _, amp = self.amp.points()
        _, phase = self.phase.points()
        _, bw = self.bw.points()
        return np.column_stack((time, freq, amp, phase, bw))
        
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
    
    def intersects(self, t):
        return self.t0 <= t <= self.t1

    def resample(self, dt):
        N = (self.t1 - self.t0) / dt
        times = np.linspace(self.t0, self.t1, N)
        f = self.freq.map(times)
        a = self.amp.map(times)
        ph = self.phase.map(times)
        bw = self.bw.map(times)
        return Partial(self.id, times, f, a, ph, bw)

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

_SpectrumFilter = _namedtuple("SpectrumFilter", "selected rejected")

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

        partials: a seq. of Partial

        See Also:

        fromarray, io.fromsdif, io.fromcsv, io.fromtxt, io.fromhdf5
        """
        assert all(isinstance(partial, Partial) for partial in partials)
        self.partials = list(partials) if not isinstance(partials, list) else partials
        self.partials.sort(key=lambda p:p.t0)
        self.reset()
        
    def reset(self):
        self.t0 = min(p.t0 for p in self.partials) if self.partials else 0
        self.t1 = max(p.t1 for p in self.partials) if self.partials else 0
        
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
        data2 = [(f2m(f), amp2db(a)) for a, f in data if a >= minamp]
        out = _music.Notes(data2)
        return out

    def filter_quick(self, mindur=0, minamp= -90, minfreq=0, maxfreq=24000):
        """
        intended for a quick filtering of undesired partials

        Returns a new Spectrum with the partials satisfying the given conditions

        SEE ALSO: filter
        """
        out = []
        for p in self.partials:
            if p.duration > mindur and p.meanamp > minamp and (minfreq <= p.meanfreq < maxfreq):
                out.append(p)
        return Spectrum(out)

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
            minamp = freqcurve_amp( p.meanfreq_weighted )
            if durcurve:
                dur_minamp = durcurve_amp( p.duration )
                minamp = min(minamp, dur_minamp)
            if p.meanamp < minamp:
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
        from . import io
        outfile = 'tmp.txt'
        self.write(outfile)
        _call_spear_with(outfile)

    def write(self, outfile, **options):
        """
        write the partial information as

        .txt  : in the format defined by spear
        .sdif : SDIF format. 
                Options:
                    rbep (bool): if True, save in RBEP format (default)
                                 if False, save in 1TRC format.
        .csv  : dump all breakpoints to csv with columns 
                [partial-id, time, freq, amp]
        .hdf5 : use a HDF5 based format
        """
        from . import io
        func = {
            '.txt' : io.tospear,
            '.hdf5': io.tohdf5,
            '.h5'  : io.tohdf5,
            '.sdif': io.tosdif,
            '.csv' : io.tocsv
        }
        base, ext = os.path.splitext(outfile)
        if ext in func:
            return func[ext](self, outfile, **options)
        else:
            raise ValueError("Format not supported")

    def writesdif(self, outfile, rbep=True, fadetime=0):
        """
        Write this spectrum to sdif

        rbep: saves the data as is, does not resample to 
              a common timegrid. This format is recommended
              over 1TRC if your software supports it
              If False, the partials are resampled to fit to 
              a common timegrid. They are saved in the 1TRC
              format.

        fadetime: if > 0, a fade-in or fade-out will be added
                  (with the given duration) for the partials 
                  which either start or end with a breakpoint
                  with an amplitude higher than 0
        """
        from . import io
        return io.tosdif(self, outfile, rbep=rbep, fadetime=fadetime)

    def toarray(self):
        """
        Returns a tuple (matrices, labels), where matrices is an iterator of
        matrix --> 2D array with columns [time freq amp phase bw]
        """
        matrices = (partial.toarray() for partial in self)
        labels = [partial.id for partial in self]
        return matrices, labels

    def resample(self, dt):
        """
        Returns a new Spectrum, resampled using dt as sampling period
        """
        partials = [p.resample(dt) for p in self.partials]
        return Spectrum(partials)

        
    # <--------------------------------- END Spectrum

def fromarray(partials):
    """
    construct a Spectrum from array data

    partials: a seq. of (label, matrix), where matrix is a 2D array with columns
              [time, freq, amp, phase, bw]
    """
    partial_list = []
    for label, matrix in partials:
        times = matrix[:,0]
        if len(times) > 1 and times[-1] - times[0] > 0:
            freq = matrix[:, 1]
            amp = matrix[:, 2]
            phase = matrix[:, 3]
            bw = matrix[:, 4]
            partial = Partial(label, times, freq, amp, phase, bw)
            partial_list.append(partial)
        else:
            _logger.debug("skipping short partial")
    return Spectrum(partial_list)

def merge(*spectra):
    partials = []
    for spectrum in spectra:
        partials.extend(spectrum.partials)
    s = Spectrum(partials)
    return s


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

