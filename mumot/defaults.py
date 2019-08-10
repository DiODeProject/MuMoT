from typing import Tuple


class MuMoTdefault:
    """Store default parameters."""

    _initialRateValue = 2  # @todo: was 1 (choose initial values sensibly)
    _rateLimits = (0.0, 20.0)  # @todo: choose limit values sensibly
    _rateStep = 0.1  # @todo: choose rate step sensibly

    @staticmethod
    def setRateDefaults(initRate=_initialRateValue,
                        limits: Tuple[float, float] = _rateLimits,
                        step: float = _rateStep) -> None:
        MuMoTdefault._initialRateValue = initRate
        MuMoTdefault._rateLimits = limits
        MuMoTdefault._rateStep = step

    _maxTime = 3
    _timeLimits = (0, 10)
    _timeStep = 0.1
    # _maxTime = 5
    # _timeLimits = (0, 50)
    # _timeStep = 0.5

    @staticmethod
    def setTimeDefaults(initTime=_maxTime, limits=_timeLimits,
                        step=_timeStep) -> None:
        MuMoTdefault._maxTime = initTime
        MuMoTdefault._timeLimits = limits
        MuMoTdefault._timeStep = step

    _agents = 1.0
    _agentsLimits = (0.0, 1.0)
    _agentsStep = 0.01

    @staticmethod
    def setAgentsDefaults(initAgents: float = _agents,
                          limits: Tuple[float, float] = _agentsLimits,
                          step: float = _agentsStep) -> None:
        MuMoTdefault._agents = initAgents
        MuMoTdefault._agentsLimits = limits
        MuMoTdefault._agentsStep = step

    _systemSize = 10
    _systemSizeLimits = (5, 100)
    _systemSizeStep = 1

    @staticmethod
    def setSystemSizeDefaults(initSysSize: int = _systemSize,
                              limits: Tuple[int, int] = _systemSizeLimits,
                              step: int = _systemSizeStep) -> None:
        MuMoTdefault._systemSize = initSysSize
        MuMoTdefault._systemSizeLimits = limits
        MuMoTdefault._systemSizeStep = step

    _plotLimits = 1.0
    _plotLimitsLimits = (0.1, 5.0)
    _plotLimitsStep = 0.1

    @staticmethod
    def setPlotLimitsDefaults(initPlotLimits: float = _plotLimits,
                              limits: Tuple[float, float] = _plotLimitsLimits,
                              step: float = _plotLimitsStep) -> None:
        MuMoTdefault._plotLimits = initPlotLimits
        MuMoTdefault._plotLimitsLimits = limits
        MuMoTdefault._plotLimitsStep = step
