import logging as _logging

_logging.basicConfig(format = ">> %(message)s")
_logger = _logging.getLogger()
_logger.setLevel(_logging.WARN)  # leave this line uncommented to see only warnings

def get_logger():
    return _logger