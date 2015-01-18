from peach import *
import warnings as _warnings

try:
    from em import comp as _comp
    _COMP_AVAILABLE = True
except ImportError:
    _warnings.warn("could not import em.comp, using simple music backend")
    _COMP_AVAILABLE = False


def _normalizenote(note):
    """
    note can be a tuple (midinote, dbamp) or a midinote

    Returns: (midinote, dbamp)
    """
    if isinstance(note, (tuple, list)) and len(note) == 2:
        return note
    return (note, 0)


class _Chord(object):
    def __init__(self, notes):
        """
        a note is a midinote or a tuple (midinote, amp_in_dbs)
        """
        self.notes = map(_normalizenote, notes)
        
    def __repr__(self):
        lines = []
        for n in self.notes:
            midinote, amp = n
            pitchstr = m2n(midinote).ljust(6)
            l = "%s | %d" % (pitchstr, int(amp))
            lines.append(l)
        return "\n".join(lines)


def partial2m21(partial, pitchres=0.5):
    """
    convert partial to a music21 stream

    pitchres: the resolution to quantize pitches
    """
    pass


def newchord(data):
    """
    data: a seq. of tuples (midinote, amp)
    """
    if _COMP_AVAILABLE:
        return _comp.music.Chord(data)
    else:
        return _Chord(data)