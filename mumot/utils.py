import math
import numbers
from typing import Optional, List

import numpy as np
from sympy.parsing.latex import parse_latex
from warnings import warn

from . import (
    consts,
    defaults,
    exceptions,
    utils,
    __version__
)


def about() -> None:
    """Print version, author and documentation information."""
    print("Multiscale Modelling Tool (MuMoT): Version " + __version__)
    print("Authors: James A. R. Marshall, Andreagiovanni Reina, Thomas Bose")
    print("Contributors: Robert Dennison, Will Furnass")
    print("Documentation: https://mumot.readthedocs.io/")


def _greekPrependify(s: str) -> str:
    """Prepend two backslash symbols in front of Greek letters to enable proper LaTeX rendering."""
    for i, letter in enumerate(consts.GREEK_LETT_LIST_1):
        if 'eta' in s:
            s = _greekReplace(s, 'eta', '\\eta')
        if letter in s:
            s = _greekReplace(s, letter, consts.GREEK_LETT_LIST_2[i])
            # if s[s.find(letter+'_')-1] !='\\':
            #    s = s.replace(letter+'_',GREEK_LETT_LIST_2[i]+'_')
    return s


def _greekReplace(s: str, sub: str, repl: str) -> str:
    """Auxiliary function for _greekPrependify()."""
    # if find_index is not minus1 we have found at least one match for the substring
    find_index = s.find(sub)
    # loop util we find no (more) match
    while find_index != -1:
        if find_index == 0 or (s[find_index - 1] != '\\' and not s[find_index - 1].isalpha()):
            if sub != 'eta':
                s = s[:find_index] + repl + s[find_index + len(sub):]
            else:
                if s[find_index - 1] != 'b' and s[find_index - 1] != 'z':
                    if s[find_index - 1] != 'h':
                        s = s[:find_index] + repl + s[find_index + len(sub):]
                    elif s[find_index - 1] == 'h':
                        if s[find_index - 2] != 't' and s[find_index - 2] != 'T':
                            s = s[:find_index] + repl + s[find_index + len(sub):]
        # find + 1 means we start at the last match start index + 1
        find_index = s.find(sub, find_index + 1)
    return s


def _doubleUnderscorify(s: str) -> str:
    """Set underscores in expressions which need two indices to enable proper LaTeX rendering."""
    ind_list = [kk for kk, char in enumerate(s)
                if char == '_' and s[kk + 1] != '{']
    if len(ind_list) == 0:
        return s
    else:
        index_MinCharLength = 1
        index_MaxCharLength_init = 20
        s_list = list(s)

        for ind in ind_list:
            ind_diff = len(s_list) - 1 - ind
            if ind_diff > 5:
                index_MaxCharLength = min(index_MaxCharLength_init, ind_diff - 5)
                # the following requires that indices consist of 1 or 2 charcter(s) only
                for nn in range(4 + index_MinCharLength, 5 + index_MaxCharLength):
                    if s_list[ind + nn] == '}' and s_list[ind + nn + 1] != '}':
                        s_list[ind] = '_{'
                        s_list[ind + nn] = '}}'
                        break

    return ''.join(s_list)


def _count_sig_decimals(digits: str, maximum: Optional[int] = 7) -> int:
    """Return the number of significant decimals of the input digit string (up to a maximum of 7)."""
    _, _, fractional = digits.partition(".")

    if fractional:
        return min(len(fractional), maximum)
    else:
        return 0


def _format_advanced_option(optionName: str, inputValue, initValues, extraParam=None, extraParam2=None):
    """Check if the user-specified values are within valid range (appropriate subfunctions are called depending on the parameter).

    parameters for slider widgets return list of length 5 as [value, min, max, step, fixed]

    parameters for boolean, dropbox, or input fields return list of lenght two as [value, fixed]

    values is the initial value, (min,max,step) are for sliders, and fixed is a boolean that indciates if the parameter is fixed or the widget should be displayed

    """
    if optionName == 'initialState':
        (allReactants, _) = extraParam
        #fixSumTo1 = extraParam2[0] # until we have better information, all views should sum to 1, then use system size to scale
        fixSumTo1 = True
        idleReactant = extraParam2[1]
        initialState = {}
        # handle initialState dictionary (either convert or generate a default one)
        if inputValue is not None:
            for reactant in sorted(inputValue.keys(), key=str):
                pop = inputValue[reactant]
                initPop = initValues.get(reactant) if initValues is not None else None

                # Convert string into SymPy symbol
                initialState[parse_latex(reactant)] = _parse_input_keyword_for_numeric_widgets(
                    inputValue=pop,
                    defaultValueRangeStep=[defaults.MuMoTdefault._agents,
                                           defaults.MuMoTdefault._agentsLimits[0],
                                           defaults.MuMoTdefault._agentsLimits[1],
                                           defaults.MuMoTdefault._agentsStep],
                    initValueRangeStep=initPop,
                    validRange=(0.0, 1.0) if fixSumTo1 else (0, float("inf")))
                fixedBool = True
        else:
            first = True
            initValuesSympy = ({parse_latex(reactant): pop
                                for reactant, pop in initValues.items()}
                               if initValues is not None else {})
            for reactant in sorted(allReactants, key=str):
                defaultV = defaults.MuMoTdefault._agents if first else 0
                first = False
                initialState[reactant] = _parse_input_keyword_for_numeric_widgets(
                    inputValue=None,
                    defaultValueRangeStep=[defaultV,
                                           defaults.MuMoTdefault._agentsLimits[0],
                                           defaults.MuMoTdefault._agentsLimits[1],
                                           defaults.MuMoTdefault._agentsStep],
                    initValueRangeStep=initValuesSympy.get(reactant),
                    validRange=(0.0, 1.0) if fixSumTo1 else (0, float("inf")))
                fixedBool = False

        # Check if the initialState values are valid
        if fixSumTo1:
            sumValues = sum([initialState[reactant][0] for reactant in allReactants])
            minStep = min([initialState[reactant][3] for reactant in allReactants])
            
            # first thing setting the values of the idleReactant
            if idleReactant is not None: 
                idleValue = initialState[idleReactant][0]
                if idleValue > 1: 
                    wrn_msg = f"WARNING! the initial value of reactant {idleReactant} has been changed to {new_val}\n"
                    warn(wrn_msg, exceptions.MuMoTWarning)
                    initialState[idleReactant][0] = new_val
                # the idleValue have range min-max reset to [0,1]
                initialState[idleReactant][1] = 0
                initialState[idleReactant][2] = 1
                initialState[idleReactant][3] = minStep
            else:
                idleValue = 0
            for reactant in sorted(allReactants, key=str):
                if reactant not in allReactants:
                    error_msg = (f"Reactant '{reactant}' does not exist in this model.\n"
                                 f"Valid reactants are {allReactants}. Please, correct the value and retry.")
                    raise exceptions.MuMoTValueError(error_msg)
    
                # check if the proportions sum to 1
                if reactant != idleReactant:
                    pop = initialState[reactant]
                    # modify (if necessary) the initial value
                    if sumValues > 1:
                        new_val = max(0, pop[0] + (1 - sumValues))
                        if not _almostEqual(pop[0], new_val):
                            wrn_msg = f"WARNING! the initial value of reactant {reactant} has been changed to {new_val}\n"
                            warn(wrn_msg, exceptions.MuMoTWarning)
                            sumValues -= pop[0]
                            sumValues += new_val
                            initialState[reactant][0] = new_val
                    # modify (if necessary) min-max
                    if idleReactant is not None:
                        pop = initialState[reactant]
                        sumNorm = sumValues if sumValues <= 1 else 1
                        if pop[2] > (1 - sumNorm + pop[0] + idleValue):  # max
                            if pop[1] > (1 - sumNorm + pop[0] + idleValue):  # min
                                initialState[reactant][1] = (1 - sumNorm + pop[0] + idleValue)
                            initialState[reactant][2] = (1 - sumNorm + pop[0] + idleValue)
                        if pop[1] > (1 - sumNorm + pop[0]):  # min
                            initialState[reactant][1] = (1 - sumNorm + pop[0])
                        # initialState[reactant][3] = minStep
            if not _almostEqual(sumValues, 1):
                reactantToFix = sorted(allReactants, key=str)[0] if idleReactant is None else idleReactant
                new_val = 1 - sum([initialState[reactant][0]
                                   for reactant in allReactants
                                   if reactant != reactantToFix])
                wrn_msg = f"WARNING! the initial value of reactant {reactantToFix} has been changed to {new_val}\n"
                warn(wrn_msg, exceptions.MuMoTWarning)
                initialState[reactantToFix][0] = new_val
        return [initialState, fixedBool]
        # print("Initial State is " + str(initialState))
    if optionName == 'maxTime':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=([2, 0.1, 3, 0.1]
                                   if extraParam == 'asNoise'
                                   else [defaults.MuMoTdefault._maxTime,
                                         defaults.MuMoTdefault._timeLimits[0],
                                         defaults.MuMoTdefault._timeLimits[1],
                                         defaults.MuMoTdefault._timeStep]),
            initValueRangeStep=initValues,
            validRange=(0, float("inf")))
    if optionName == 'randomSeed':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=np.random.randint(consts.MAX_RANDOM_SEED),
            initValueRangeStep=initValues,
            validRange=(1, consts.MAX_RANDOM_SEED), onlyValue=True)
    if optionName == 'motionCorrelatedness':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=[0.5, 0.0, 1.0, 0.05],
            initValueRangeStep=initValues,
            validRange=(0, 1))
    if optionName == 'particleSpeed':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=[0.01, 0.0, 0.1, 0.005],
            initValueRangeStep=initValues,
            validRange=(0, 1))

    if optionName == 'timestepSize':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=[1, 0.01, 1, 0.01],
            initValueRangeStep=initValues,
            validRange=(0, float("inf")))

    if optionName == 'netType':
        # check validity of the network type or init to default
        if inputValue is not None:
            decodedNetType = utils._decodeNetworkTypeFromString(inputValue)
            if decodedNetType is None:  # terminating the process if the input argument is wrong
                error_msg = (f"The specified value for netType ={inputValue} is not valid. \n"
                             "Accepted values are: 'full',  'erdos-renyi', 'barabasi-albert', and 'dynamic'.")
                raise exceptions.MuMoTValueError(error_msg)

            return [inputValue, True]
        else:
            decodedNetType = utils._decodeNetworkTypeFromString(initValues) if initValues is not None else None
            if decodedNetType is not None:  # assigning the init value only if it's a valid value
                return [initValues, False]
            else:
                return ['full', False]  # as default netType is set to 'full'
    # @todo: avoid that these value will be overwritten by _update_net_params()
    if optionName == 'netParam':
        netType = extraParam
        systemSize = extraParam2
        # if netType is not fixed, netParam cannot be fixed
        if (not netType[-1]) and inputValue is not None:
            error_msg = ("If netType is not fixed, netParam cannot be fixed. "
                         "Either leave free to widget the 'netParam' or fix the 'netType'.")
            raise exceptions.MuMoTValueError(error_msg)
        # check if netParam range is valid or set the correct default range (systemSize is necessary)
        if utils._decodeNetworkTypeFromString(netType[0]) == consts.NetworkType.FULLY_CONNECTED:
            return [0, 0, 0, False]
        elif utils._decodeNetworkTypeFromString(netType[0]) == consts.NetworkType.ERSOS_RENYI:
            return _parse_input_keyword_for_numeric_widgets(
                inputValue=inputValue,
                defaultValueRangeStep=[0.1, 0.1, 1, 0.1],
                initValueRangeStep=initValues,
                validRange=(0.1, 1.0))
        elif utils._decodeNetworkTypeFromString(netType[0]) == consts.NetworkType.BARABASI_ALBERT:
            maxEdges = systemSize - 1
            return _parse_input_keyword_for_numeric_widgets(
                inputValue=inputValue,
                defaultValueRangeStep=[min(maxEdges, 3), 1, maxEdges, 1],
                initValueRangeStep=initValues,
                validRange=(1, maxEdges))
        elif utils._decodeNetworkTypeFromString(netType[0]) == consts.NetworkType.SPACE:
            pass  # method is not implemented
        elif utils._decodeNetworkTypeFromString(netType[0]) == consts.NetworkType.DYNAMIC:
            return _parse_input_keyword_for_numeric_widgets(
                inputValue=inputValue,
                defaultValueRangeStep=[0.1, 0.0, 1.0, 0.05],
                initValueRangeStep=initValues,
                validRange=(0, 1.0))
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=[0.5, 0, 1, 0.1],
            initValueRangeStep=initValues,
            validRange=(0, float("inf")))

    if optionName == 'plotProportions':
        return _parse_input_keyword_for_boolean_widgets(
            inputValue=inputValue,
            defaultValue=False,
            initValue=initValues,
            paramNameForErrorMsg=optionName)
    if optionName == 'realtimePlot':
        return _parse_input_keyword_for_boolean_widgets(
            inputValue=inputValue,
            defaultValue=False,
            initValue=initValues,
            paramNameForErrorMsg=optionName)
    if optionName == 'showTrace':
        return _parse_input_keyword_for_boolean_widgets(
            inputValue=inputValue,
            defaultValue=False,
            initValue=initValues,
            paramNameForErrorMsg=optionName)
    if optionName == 'showInteractions':
        return _parse_input_keyword_for_boolean_widgets(
            inputValue=inputValue,
            defaultValue=False,
            initValue=initValues,
            paramNameForErrorMsg=optionName)

    if optionName == 'visualisationType':
        if extraParam is not None:
            if extraParam == 'multiagent':
                validVisualisationTypes = ['evo', 'graph', 'final', 'barplot']
            elif extraParam == "SSA":
                validVisualisationTypes = ['evo', 'final', 'barplot']
            elif extraParam == "multicontroller":
                validVisualisationTypes = ['evo', 'final']
        else:
            validVisualisationTypes = ['evo', 'graph', 'final']
        if inputValue is not None:
            if inputValue not in validVisualisationTypes:  # terminating the process if the input argument is wrong
                errorMsg = (f"The specified value for visualisationType = {inputValue} is not valid.\n"
                            f"Valid values are: {validVisualisationTypes}. Please correct it and retry.")
                raise exceptions.MuMoTValueError(errorMsg)
            return [inputValue, True]
        else:
            if initValues in validVisualisationTypes:
                return [initValues, False]
            else:
                return ['evo', False]  # as default visualisationType is set to 'evo'

    if optionName in ('final_x', 'final_y'):
        reactants_str = [str(reactant) for reactant in sorted(extraParam, key=str)]
        if inputValue is not None:
            inputValue = inputValue.replace('\\', '')
            if inputValue not in reactants_str:
                error_msg = (f"The specified value for {optionName} = {inputValue} is not valid.\n"
                             f"Valid values are the reactants: {reactants_str}. Please correct it and retry.")
                raise exceptions.MuMoTValueError(error_msg)
            else:
                return [inputValue, True]
        else:
            if initValues is not None:
                initValues = initValues.replace('\\', '')
            if initValues in reactants_str:
                return [initValues, False]
            else:
                if optionName == 'final_x' or len(reactants_str) == 1:
                    return [reactants_str[0], False]  # as default final_x is set to the first (sorted) reactant
                else:
                    return [reactants_str[1], False]  # as default final_y is set to the second (sorted) reactant

    if optionName == 'runs':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=([20, 5, 100, 5]
                                   if extraParam == 'asNoise'
                                   else [1, 1, 20, 1]),
            initValueRangeStep=initValues,
            validRange=(1, float("inf")))

    if optionName == 'aggregateResults':
        return _parse_input_keyword_for_boolean_widgets(
            inputValue=inputValue,
            defaultValue=True,
            initValue=initValues,
            paramNameForErrorMsg=optionName)

    if optionName == 'initBifParam':
        return _parse_input_keyword_for_numeric_widgets(
            inputValue=inputValue,
            defaultValueRangeStep=[defaults.MuMoTdefault._initialRateValue,
                                   defaults.MuMoTdefault._rateLimits[0],
                                   defaults.MuMoTdefault._rateLimits[1],
                                   defaults.MuMoTdefault._rateStep],
            initValueRangeStep=initValues,
            validRange=(0, float("inf")))

    return [None, False]  # default output for unknown optionName


def _get_item_from_params_list(params: List[str], targetName: str) -> Optional[str]:
    """Params is a list (rather than a dictionary) and this method is necessary to fetch the value by name."""
    for param in params:
        if param[0] == targetName or param[0].replace('\\', '') == targetName or param[0].replace('_', '_{') + '}' == targetName:
            return param[1]
    return None


def _almostEqual(a: float, b: float) -> bool:
    epsilon = 0.0000001
    return abs(a - b) < epsilon


def _parse_input_keyword_for_numeric_widgets(
        inputValue: Optional[object],
        defaultValueRangeStep: List[object],
        initValueRangeStep: List[object],
        validRange: Optional[List[object]] = None,
        onlyValue: Optional[bool] = False) -> List[object]:
    """Parse an input keyword and set initial range and default values (when the input is a slider-widget).

    Check if the fixed value is not None, otherwise it returns the default value (samewise for ``initRange`` and ``defaultRange``).
    The optional parameter ``validRange`` is use to check if the fixedValue has a usable value.
    If the ``defaultValue`` is out of the ``initRange``, the default value is move to the closest of the initRange extremes.

    Parameters
    ----------
    inputValue : object
        if not ``None`` it indicated the fixed value to use
    defaultValueRangeStep : list of object
        Default set of values in the format ``[val,min,max,step]``
    initValueRangeStep : list of object
        User-specified set of values in the format ``[val,min,max,step]``
    validRange : list of object, optional
        The min and max accepted values ``[min,max]``
    onlyValue : bool, optional
        If ``True`` then ``defaultValueRangeStep`` and ``initValueRangeStep`` are only a single value

    Returns
    -------
    values : list of object
        Contains a list of five items (start-value, min-value, max-value,
        step-size, fixed). if onlyValue, it's only two items (start-value,
        fixed). The item fixed is a bool. If ``True`` the value is fixed (partial
        controller active), if ``False`` the widget will be created.

    """
    outputValues = defaultValueRangeStep if not onlyValue else [defaultValueRangeStep]
    if not onlyValue:
        if initValueRangeStep is not None and getattr(initValueRangeStep, "__getitem__", None) is None:
            error_msg = (f"initValueRangeStep value '{initValueRangeStep}' must be specified in the format [val,min,max,step].\n"
                         "Please, correct the value and retry.")
            raise exceptions.MuMoTValueError(error_msg)
    if inputValue is not None:
        if not isinstance(inputValue, numbers.Number):
            error_msg = (f"Input value '{inputValue}' is not a numeric vaule and must be a number.\n"
                         "Please, correct the value and retry.")
            raise exceptions.MuMoTValueError(error_msg)
        elif validRange and (inputValue < validRange[0] or inputValue > validRange[1]):
            error_msg = (f"Input value '{inputValue}' has raised out-of-range exception. Valid range is {validRange}\n"
                         "Please, correct the value and retry.")
            raise exceptions.MuMoTValueError(error_msg)
        else:
            if onlyValue:
                return [inputValue, True]
            else:
                outputValues[0] = inputValue
                outputValues.append(True)
                # it is not necessary to modify the values [min,max,step] because when last value is True, they should be ignored
                return outputValues

    if initValueRangeStep is not None:
        if onlyValue:
            if validRange and (initValueRangeStep < validRange[0] or initValueRangeStep > validRange[1]):
                error_msg = (f"Invalid init value={initValueRangeStep} has raised out-of-range exception. Valid range is {validRange}\n"
                             "Please, correct the value and retry.")
                raise exceptions.MuMoTValueError(error_msg)
            else:
                outputValues = [initValueRangeStep]
        else:
            if initValueRangeStep[1] > initValueRangeStep[2] or initValueRangeStep[0] < initValueRangeStep[1] or initValueRangeStep[0] > initValueRangeStep[2]:
                error_msg = (f"Invalid init range [val,min,max,step]={initValueRangeStep}. Value must be within min and max values.\n"
                             "Please, correct the value and retry.")
                raise exceptions.MuMoTValueError(error_msg)
            elif validRange and (initValueRangeStep[1] < validRange[0] or initValueRangeStep[2] > validRange[1]):
                error_msg = (f"Invalid init range [val,min,max,step]={initValueRangeStep} has raised out-of-range exception. Valid range is {validRange}\n"
                             "Please, correct the value and retry.")
                raise exceptions.MuMoTValueError(error_msg)
            else:
                outputValues = initValueRangeStep

    outputValues.append(False)
    return outputValues


def _parse_input_keyword_for_boolean_widgets(inputValue, defaultValue, initValue=None, paramNameForErrorMsg=None):
    """Parse an input keyword and set initial range and default values (when the input is a boolean checkbox)
    check if the fixed value is not None, otherwise it returns the default value.

    Parameters
    ----------
    inputValue : object
        If not None it indicates the fixed value to use
    defaultValue : bool
        dafault boolean value

    Returns
    -------
    value : object
        The keyword value; 'fixed' is a boolean.
    fixed : bool
        If True the value is fixed (partial controller active); if False the widget will be created.

    """
    if inputValue is not None:
        if not isinstance(inputValue, bool):  # terminating the process if the input argument is wrong
            paramNameForErrorMsg = f"for {paramNameForErrorMsg} = " if paramNameForErrorMsg else ""
            errorMsg = (f"The specified value {paramNameForErrorMsg}'{inputValue}' is not valid. \n"
                        "The value must be a boolean True/False.")
            raise exceptions.MuMoTValueError(errorMsg)
        return [inputValue, True]
    else:
        if isinstance(initValue, bool):
            return [initValue, False]
        else:
            return [defaultValue, False]


def _process_params(params):
    paramsRet = []
    paramNames, paramValues = zip(*params)
    for name in paramNames:
        # self._paramNames.append(name.replace('\\','')) ## @todo: have to rationalise how LaTeX characters are handled
        if name in ('plotLimits', 'systemSize'):
            paramsRet.append(name)
        else:
            expr = parse_latex(name.replace('\\\\', '\\'))
            atoms = expr.atoms()
            if len(atoms) > 1:
                raise exceptions.MuMoTSyntaxError(f"Non-singleton parameter name in parameter {name}")
            for atom in atoms:
                # parameter name should contain a single atom
                pass
            paramsRet.append(atom)

    return (paramsRet, paramValues)


def _decodeNetworkTypeFromString(netTypeStr: str) -> Optional[consts.NetworkType]:
    # init the network type
    admissibleNetTypes = {'full': consts.NetworkType.FULLY_CONNECTED,
                          'erdos-renyi': consts.NetworkType.ERSOS_RENYI,
                          'barabasi-albert': consts.NetworkType.BARABASI_ALBERT,
                          'dynamic': consts.NetworkType.DYNAMIC}

    if netTypeStr not in admissibleNetTypes:
        raise exceptions.MuMoTValueError(f"ERROR! Invalid network type argument! Valid strings are: {admissibleNetTypes}")
    return admissibleNetTypes.get(netTypeStr, None)


def _encodeNetworkTypeToString(netType: consts.NetworkType) -> Optional[str]:
    # init the network type
    netTypeEncoding = {consts.NetworkType.FULLY_CONNECTED: 'full',
                       consts.NetworkType.ERSOS_RENYI: 'erdos-renyi',
                       consts.NetworkType.BARABASI_ALBERT: 'barabasi-albert',
                       consts.NetworkType.DYNAMIC: 'dynamic'}

    if netType not in netTypeEncoding:
        raise exceptions.MuMoTValueError(f"ERROR! Invalid netTypeEncoding table! Tried to encode network type: {netType}")
    return netTypeEncoding.get(netType, 'none')


def _round_to_1(x):
    """Used for determining significant digits for axes formatting in plots MuMoTstreamView and MuMoTbifurcationView."""
    if x == 0:
        return 1
    return round(x, -int(math.floor(math.log10(abs(x)))))


def _make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
    return my_autopct
