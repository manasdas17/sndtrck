SNDTRCK
=======

A simple data-type and io routines for audio partial tracking

# Dependencies

## Mandatory

* [numpy]  
* [bpf4]   -- interpolation curves
* [peach]  -- for pitch and note conversion

## Optional but recommended

* [loristrck] -- partial tracking analysis based on Loris
* [pandas]    -- allows saving partial-tracking data as HDF5
* [sndfileio] -- simple API for reading and writing sound-files

# Installation

    $ git clone https://github.com/gesellkammer/sndtrck
    $ cd sndtrck
    $ python setup.py install

# Usage

    TODO

[bpf4]: https://github.com/gesellkammer/bpf4
[peach]: https://github.com/gesellkammer/peach
[loristrck]: https://github.com/gesellkammer/loristrck
[sndfileio]: https://github.com/gesellkammer/sndfileio
[pandas]: http://pandas.pydata.org/