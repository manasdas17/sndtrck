class Notes(object):
    def __init__(self, notes):
        """
        a note is a midinote or a tuple (midinote, amp_in_dbs)
        """
        _notes = []
        for n in notes:
            if isinstance(n, (tuple, list)):
                _notes.append(n)
            else:
                _notes.append((n, 0))
        self.notes = _notes
        
    def __repr__(self):
        lines = []
        for n in self.notes:
            midinote, amp = n
            pitchstr = m2n(midinote).ljust(6)
            l = "%s | %d" % (pitchstr, int(amp))
            lines.append(l)
        return "\n".join(lines)

