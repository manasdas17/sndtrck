import bpf4

def bpf2partial(freq, amp=None, dt=None):
    """
    Create a Partial from a bpf representing
    frequency and a number or bpf representing amplitude

    freq: a bpf representing a frequency curve
    amp: a bpf or constant representing an envelope
    dt: the sampling period to sample the curves

    Example:

    midi = bpf4.linear(0, 60, 10, 72)
    freq = midi.m2f()
    partial = bpf2partial(freq, 0.5)
    Spectrum([partial]).show()
    """
    f = bpf4.asbpf(freq)
    a = bpf4.asbpf(amp)
    x0 = max(f.bounds()[0], a.bounds()[0])
    x1 = min(f.bounds()[1], a.bounds()[1])
    if dt is None:
        xs, ys = f._get_points_for_rendering()
        N = len(xs)
        x0, x1 = f.bounds()
        dt = (x1 - x0)/N
    else:
        N = int((x1-x0)/dt)
    times = np.linspace(x0, x1, N)
    freqs = f.mapn_between(N, x0, x1)
    amps = a.map(N)
    assert len(times) == len(freqs) == len(amps)
    p = sndtrck.Partial(0, times, freqs, amps)
    return p

    