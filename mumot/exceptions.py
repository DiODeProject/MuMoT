"""MuMoT warning, exception and error classes."""

class MuMoTWarning(Warning):
    """Class to report MuMoT-specific warnings.
    """
    pass


class MuMoTError(Exception):
    """Class to report MuMoT-specific errors.
    """
    pass


class MuMoTValueError(MuMoTError):
    """Class to report MuMoT-specific errors arising from incorrect input.
    """
    pass


class MuMoTSyntaxError(MuMoTError):
    """Class to report MuMoT-specific errors arising from incorrectly-structured input.
    """
    pass


def _raiseModelError(expected, read, rule):
    raise MuMoTSyntaxError(f"Expected {expected} but read '{read}' in rule: {rule}")
