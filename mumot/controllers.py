"""MuMoT controller classes."""

import base64

from IPython.display import Javascript, display
import ipywidgets.widgets as widgets
from ipywidgets import HTML
from sympy import Symbol

from . import (
    consts,
    defaults,
    exceptions,
    utils,
    views,
)


class MuMoTcontroller:
    """Controller for a model view."""

    _view = None
    # dictionary of LaTeX labels for widgets, with parameter name as key
    _paramLabelDict = None
    # dictionary of controller widgets only for the free parameters of the model, with parameter name as key
    _widgetsFreeParams = None
    # dictionary of controller widgets for the special parameters (e.g., simulation length, initial state), with parameter name as key
    _widgetsExtraParams = None
    # dictionary of controller widgets, with parameter that influence only the plotting and not the computation
    _widgetsPlotOnly = None
    # list keeping the order of the extra-widgets (_widgetsExtraParams and _widgetsPlotOnly)
    _extraWidgetsOrder = None
    # replot function widgets have been assigned (for use by MuMoTmultiController)
    _replotFunction = None
    # redraw function widgets have been assigned (for use by MuMoTmultiController)
    _redrawFunction = None
    # widget for simple error messages to be displayed to user during interaction
    _errorMessage = None
    # plot limits slider widget
    _plotLimitsWidget = None  # @todo: is it correct that this is a variable of the general MuMoTcontroller?? it might be simply added in the _widgetsPlotOnly dictionary
    # system size slider widget
    _systemSizeWidget = None
    # bookmark button widget
    _bookmarkWidget = None
    # advanced tab widget
    _advancedTabWidget = None
    # download button widget
    _downloadWidget = None
    # download link widget
    _downloadWidgetLink = None

    def __init__(self, paramValuesDict, paramLabelDict=None,
                 continuousReplot: bool = False, showPlotLimits: bool = False,
                 showSystemSize: bool = False, advancedOpts=None, **kwargs) -> None:
        self._silent = kwargs.get('silent', False)
        self._paramLabelDict = paramLabelDict if paramLabelDict is not None else {}
        self._widgetsFreeParams = {}
        self._widgetsExtraParams = {}
        self._widgetsPlotOnly = {}
        self._extraWidgetsOrder = []

        for paramName in sorted(paramValuesDict.keys()):
            if paramName == 'plotLimits' or paramName == 'systemSize':
                continue
            if not paramValuesDict[paramName][-1]:
                paramValue = paramValuesDict[paramName]
                widget = widgets.FloatSlider(value=paramValue[0], min=paramValue[1],
                                             max=paramValue[2], step=paramValue[3],
                                             readout_format='.' + str(utils._count_sig_decimals(str(paramValue[3]))) + 'f',
                                             description=r'\(' + utils._doubleUnderscorify(utils._greekPrependify(self._paramLabelDict.get(paramName, paramName))) + r'\)',
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                self._widgetsFreeParams[paramName] = widget
                if not self._silent:
                    display(widget)
        if showPlotLimits:
            if not paramValuesDict['plotLimits'][-1]:
                paramValue = paramValuesDict['plotLimits']
                self._plotLimitsWidget = widgets.FloatSlider(
                    value=paramValue[0], min=paramValue[1],
                    max=paramValue[2], step=paramValue[3],
                    readout_format='.' + str(utils._count_sig_decimals(str(paramValue[3]))) + 'f',
                    description='Plot limits',
                    style={'description_width': 'initial'},
                    continuous_update=continuousReplot)
                # @todo: it would be better to remove self._plotLimitsWidget and use the self._widgetsExtraParams['plotLimits'] = widget
                if not self._silent:
                    display(self._plotLimitsWidget)

        if showSystemSize:
            if not paramValuesDict['systemSize'][-1]:
                paramValue = paramValuesDict['systemSize']
                self._systemSizeWidget = widgets.FloatSlider(
                    value=paramValue[0], min=paramValue[1],
                    max=paramValue[2], step=paramValue[3],
                    readout_format='.' + str(utils._count_sig_decimals(str(paramValue[3]))) + 'f',
                    description='System size',
                    style={'description_width': 'initial'},
                    continuous_update=continuousReplot)
                if not self._silent:
                    display(self._systemSizeWidget)

        # create advanced widgets (that will be added into the 'Advanced options' tab)
        initialState = self._createAdvancedWidgets(advancedOpts, continuousReplot)
        self._orderAdvancedWidgets(initialState)
        # add widgets to the Advanced options tab
        if not self._silent:
            self._displayAdvancedOptionsTab()

        self._bookmarkWidget = widgets.Button(description='',
                                              disabled=False,
                                              button_style='',
                                              tooltip='Paste bookmark to log',
                                              icon='fa-bookmark')
        self._bookmarkWidget.on_click(self._print_standalone_view_cmd)
        bookmark = kwargs.get('bookmark', True)

        self._downloadWidget = widgets.Button(description='',
                                              disabled=False,
                                              button_style='',
                                              tooltip='Download results',
                                              icon='fa-save')
        self._downloadWidgetLink = HTML(self._create_download_link("", "", ""),
                                        visible=False)
        self._downloadWidgetLink.layout.visibility = 'hidden'
        self._downloadWidget.on_click(self._download_link_unsupported)

        if not self._silent and bookmark:
            # display(self._bookmarkWidget)

            box_layout = widgets.Layout(display='flex',
                                        flex_flow='row',
                                        align_items='stretch',
                                        width='70%')
            threeButtons = widgets.Box(children=[self._bookmarkWidget,
                                                 self._downloadWidget,
                                                 self._downloadWidgetLink],
                                       layout=box_layout)
            display(threeButtons)

        widget = widgets.HTML()
        widget.value = ''
        self._errorMessage = widget
        if not self._silent and bookmark:
            display(self._errorMessage)

    def _print_standalone_view_cmd(self, _includeParams) -> None:
        self._errorMessage.value = "Pasted bookmark to log - view with showLogs(tail = True)"
        self._view._print_standalone_view_cmd(True)

    # set the functions that must be triggered when the widgets are changed.
    # @param[in]    recomputeFunction    The function to be called when recomputing is necessary
    # @param[in]    redrawFunction    The function to be called when only redrawing (relying on previous computation) is sufficient
    def _setReplotFunction(self, recomputeFunction, redrawFunction=None) -> None:
        """set the functions that must be triggered when the widgets are changed.
        :param recomputeFunction
            The function to be called when recomputing is necessary
        :param redrawFunction
            The function to be called when only redrawing (relying on previous computation) is sufficient"""
        self._replotFunction = recomputeFunction
        self._redrawFunction = redrawFunction
        for widget in self._widgetsFreeParams.values():
            # widget.on_trait_change(recomputeFunction, 'value')
            widget.observe(recomputeFunction, 'value')
        for widget in self._widgetsExtraParams.values():
            widget.observe(recomputeFunction, 'value')
        if self._plotLimitsWidget is not None:
            self._plotLimitsWidget.observe(recomputeFunction, 'value')
        if self._systemSizeWidget is not None:
            self._systemSizeWidget.observe(recomputeFunction, 'value')
        if redrawFunction is not None:
            for widget in self._widgetsPlotOnly.values():
                widget.observe(redrawFunction, 'value')

    def _createAdvancedWidgets(self, _advancedOpts, _continuousReplot: bool = False) -> None:
        """Interface method to add advanced options (if needed)"""
        return None

    def _orderAdvancedWidgets(self, initialState) -> None:
        """Interface method to sort the advanced options, in the self._extraWidgetsOrder list"""
        pass

    def _displayAdvancedOptionsTab(self) -> None:
        """Create and display the "Advanced options" tab (if not empty)"""
        advancedWidgets = []
        atLeastOneAdvancedWidget = False
        for widgetName in self._extraWidgetsOrder:
            if self._widgetsExtraParams.get(widgetName):
                advancedWidgets.append(self._widgetsExtraParams[widgetName])
                if not self._widgetsExtraParams[widgetName].layout.display == 'none':
                    atLeastOneAdvancedWidget = True
            elif self._widgetsPlotOnly.get(widgetName):
                advancedWidgets.append(self._widgetsPlotOnly[widgetName])
                if not self._widgetsPlotOnly[widgetName].layout.display == 'none':
                    atLeastOneAdvancedWidget = True
            # else:
                # print("WARNING! In the _extraWidgetsOrder is listed the widget " + widgetName + " which is although not found in _widgetsExtraParams or _widgetsPlotOnly")
        if advancedWidgets:  # if not empty
            advancedPage = widgets.Box(children=advancedWidgets)
            advancedPage.layout.flex_flow = 'column'
            self._advancedTabWidget = widgets.Accordion(children=[advancedPage])  # , selected_index=-1)
            self._advancedTabWidget.set_title(0, 'Advanced options')
            self._advancedTabWidget.selected_index = None
            if atLeastOneAdvancedWidget:
                display(self._advancedTabWidget)

    def _setView(self, view) -> None:
        self._view = view

    def showLogs(self, tail: bool = False) -> None:
        """Show logs from associated view.

        Parameters
        ----------
        tail : bool, optional
           Flag to show only tail entries from logs. Defaults to False.
        """
        self._view.showLogs(tail)

    def _updateInitialStateWidgets(self, _=None) -> None:
        (allReactants, _) = self._view._mumotModel._getAllReactants()
        if len(allReactants) == 1:
            return
        sumNonConstReactants = 0
        for state in allReactants:
            sumNonConstReactants += self._widgetsExtraParams[f"init{state}"].value
        substitutedReactant = None
        if self._view._mumotModel._systemSize is not None:
            substitutedReactant = [react for react in allReactants
                                   if react not in self._view._mumotModel._reactants][0]
        disabledValue = 1
        for i, state in enumerate(sorted(allReactants, key=str)):
            if (substitutedReactant is None and i == 0) or (substitutedReactant is not None and state == substitutedReactant):
                disabledValue = 1 - (sumNonConstReactants - self._widgetsExtraParams[f"init{state}"].value)
                break

        for i, state in enumerate(sorted(allReactants, key=str)):
            # oder of assignment is important (first, update the min and max, later, the value)
            toLinkPlotFunction = False
            # the self._view._controller pointer is necessary to work properly with multiControllers
            # if self._replotFunction is not None:
            if self._view._controller._replotFunction is not None:
                try:
                    self._widgetsExtraParams[f"init{state}"].unobserve(self._view._controller._replotFunction, 'value')
                    # self._widgetsExtraParams['init'+str(state)].unobserve(self._replotFunction, 'value')
                    toLinkPlotFunction = True
                except ValueError:
                    pass

            disabledState = ((substitutedReactant is None and i == 0) or (substitutedReactant is not None and state == substitutedReactant))
            if disabledState:
                # print(str(state) + ": sum is " + str(sumNonConstReactants) + " - val " + str(disabledValue))
                self._widgetsExtraParams[f"init{state}"].value = disabledValue
            else:
                # maxVal = 1-disabledValue if 1-disabledValue > self._widgetsExtraParams['init'+str(state)].min else self._widgetsExtraParams['init'+str(state)].min
                maxVal = disabledValue + self._widgetsExtraParams[f"init{state}"].value
                self._widgetsExtraParams[f"init{state}"].max = maxVal

            if toLinkPlotFunction:
                self._widgetsExtraParams[f"init{state}"].observe(self._view._controller._replotFunction, 'value')

    def _updateFinalViewWidgets(self, change=None) -> None:
        if change['new'] != 'final':
            if self._widgetsPlotOnly.get('final_x'):
                self._widgetsPlotOnly['final_x'].layout.display = 'none'
            if self._widgetsPlotOnly.get('final_y'):
                self._widgetsPlotOnly['final_y'].layout.display = 'none'
        else:
            if self._widgetsPlotOnly.get('final_x'):
                self._widgetsPlotOnly['final_x'].layout.display = 'flex'
            if self._widgetsPlotOnly.get('final_y'):
                self._widgetsPlotOnly['final_y'].layout.display = 'flex'

    def _setErrorWidget(self, errorWidget) -> None:
        self._errorMessage = errorWidget

    def _downloadFileWithJavascript(self, data_to_download) -> Javascript:
        js_download = """
        var csv = '%s';

        var filename = 'results.csv';
        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        if (navigator.msSaveBlob) { // IE 10+
            navigator.msSaveBlob(blob, filename);
        } else {
            var link = document.createElement("a");
            if (link.download !== undefined) { // feature detection
                // Browsers that support HTML5 download attribute
                var url = URL.createObjectURL(blob);
                link.setAttribute("href", url);
                link.setAttribute("download", filename);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
        """ % str(data_to_download)
        # str(data_to_download)
        # data_to_download.to_csv(index=False).replace('\n','\\n').replace("'","\'")

        return Javascript(js_download)

    def _create_download_link(self, text: str, title: str = "Download file",
                              filename: str = "file.txt") -> str:
        """Create a download link."""
        b64 = base64.b64encode(text.encode())
        payload = b64.decode()
        html = '<a download="{filename}" href="data:text/text;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload, title=title, filename=filename)
        return html

    def _reveal_download_link(self, _includeParams) -> None:
        """Make download link visible."""
        self._downloadWidgetLink.layout.visibility = 'visible'

    def _download_link_unsupported(self, _includeParams) -> None:
        """Report that results download is unsupported."""
        self._view._showErrorMessage("Results download for this view is currently unsupported")


class MuMoTbifurcationController(MuMoTcontroller):
    """Controller to enable Advanced options widgets for bifurcation view."""

    def _createAdvancedWidgets(self, BfcParams, continuousReplot: bool = False):
        initialState = BfcParams['initialState'][0]
        if not BfcParams['initialState'][-1]:
            # for state,pop in initialState.items():
            for i, state in enumerate(sorted(initialState.keys(), key=str)):
                pop = initialState[state]
                widget = widgets.FloatSlider(value=pop[0],
                                             min=pop[1],
                                             max=pop[2],
                                             step=pop[3],
                                             readout_format='.' + str(utils._count_sig_decimals(str(pop[3]))) + 'f',
                                             # r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(str(stateVarExpr1))) +'}$'
                                             description=r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(self._paramLabelDict.get(state, str(state)))) + '}$' + " at t=0: ",
                                             # description = r'\(' + utils._doubleUnderscorify(utils._greekPrependify('Phi_'+self._paramLabelDict.get(state,str(state)))) + r'\)' + " at t=0: ",
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)

                if BfcParams['conserved'][0] is True:
                    # Disable one population widget (if there are more than 1) to constrain the initial sum of population sizes to 1
                    if len(initialState) > 1:
                        if BfcParams['substitutedReactant'][0] is None and i == 0:
                            # If there is not a 'substituted' reactant, the last population widget
                            widget.disabled = True
                        elif BfcParams['substitutedReactant'][0] is not None and state == BfcParams['substitutedReactant'][0]:
                            # If there is a 'substituted' reactant, this is the chosen one as the disabled pop
                            widget.disabled = True
                        else:
                            widget.observe(self._updateInitialStateWidgets, 'value')

                self._widgetsExtraParams[f"init{state}"] = widget

        # init bifurcation paramter value slider
        if not BfcParams['initBifParam'][-1]:
            initBifParam = BfcParams['initBifParam']
            widget = widgets.FloatSlider(value=initBifParam[0], min=initBifParam[1],
                                         max=initBifParam[2], step=initBifParam[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(initBifParam[3]))) + 'f',
                                         description='Initial ' + r'\(' + utils._doubleUnderscorify(utils._greekPrependify(str(BfcParams['bifurcationParameter'][0]))) + r'\) : ',
                                         style={'description_width': 'initial:'},
                                         # layout=widgets.Layout(width='50%'),
                                         disabled=False,
                                         continuous_update=continuousReplot)
            self._widgetsExtraParams['initBifParam'] = widget

        return initialState

    def _orderAdvancedWidgets(self, initialState) -> None:
        # define the widget order
        self._extraWidgetsOrder.append('initBifParam')
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append(f"init{state}")


class MuMoTtimeEvolutionController(MuMoTcontroller):
    """Controller class to enable Advanced options widgets for simulation of ODEs and time evolution of noise correlations."""

    def _createAdvancedWidgets(self, tEParams, continuousReplot=False):
        initialState = tEParams['initialState'][0]
        if not tEParams['initialState'][-1]:
            # for state,pop in initialState.items():
            for i, state in enumerate(sorted(initialState.keys(), key=str)):
                pop = initialState[state]
                widget = widgets.FloatSlider(value=pop[0],
                                             min=pop[1],
                                             max=pop[2],
                                             step=pop[3],
                                             readout_format='.' + str(utils._count_sig_decimals(str(pop[3]))) + 'f',
                                             # description = "Reactant " + r'\(' + utils._doubleUnderscorify(utils._greekPrependify(self._paramLabelDict.get(state,str(state)))) + r'\)' + " at t=0: ",
                                             description=r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(self._paramLabelDict.get(state, str(state)))) + '}$' + " at t=0: ",
                                             # description = r'\(' + latex(Symbol('Phi_'+str(state))) + r'\)' + " at t=0: ",
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)

                if tEParams['conserved'][0] is True:
                    # Disable one population widget (if there are more than 1) to constrain the initial sum of population sizes to 1
                    if len(initialState) > 1:
                        if tEParams['substitutedReactant'][0] is None and i == 0:
                            # If there is not a 'substituted' reactant, the last population widget
                            widget.disabled = True
                        elif tEParams['substitutedReactant'][0] is not None and state == tEParams['substitutedReactant'][0]:
                            # If there is a 'substituted' reactant, this is the chosen one as the disabled pop
                            widget.disabled = True
                        else:
                            widget.observe(self._updateInitialStateWidgets, 'value')

                self._widgetsExtraParams[f"init{state}"] = widget

        # Max time slider
        if not tEParams['maxTime'][-1]:
            maxTime = tEParams['maxTime']
            widget = widgets.FloatSlider(value=maxTime[0], min=maxTime[1],
                                         max=maxTime[2], step=maxTime[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(maxTime[3]))) + 'f',
                                         description='Simulation time:',
                                         style={'description_width': 'initial'},
                                         # layout=widgets.Layout(width='50%'),
                                         disabled=False,
                                         continuous_update=continuousReplot)
            self._widgetsExtraParams['maxTime'] = widget

        # Checkbox for proportions or full populations plot
        if 'plotProportions' in tEParams:
            if not tEParams['plotProportions'][-1]:
                widget = widgets.Checkbox(
                    value=tEParams['plotProportions'][0],
                    description='Plot population proportions',
                    disabled=False
                )
                self._widgetsPlotOnly['plotProportions'] = widget

        return initialState

    def _orderAdvancedWidgets(self, initialState):
        # define the widget order
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append(f"init{state}")
        self._extraWidgetsOrder.append('maxTime')
        if 'plotProportions' in self._widgetsPlotOnly:
            self._extraWidgetsOrder.append('plotProportions')


class MuMoTfieldController(MuMoTcontroller):
    """Controller for field views"""

    def _createAdvancedWidgets(self, advancedOpts, continuousReplot=False):
        # Max time slider
        if not advancedOpts['maxTime'][-1]:
            maxTime = advancedOpts['maxTime']
            widget = widgets.FloatSlider(value=maxTime[0], min=maxTime[1],
                                         max=maxTime[2], step=maxTime[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(maxTime[3]))) + 'f',
                                         description='Simulation time:',
                                         style={'description_width': 'initial'},
                                         # layout=widgets.Layout(width='50%'),
                                         disabled=False,
                                         continuous_update=continuousReplot)
            self._widgetsExtraParams['maxTime'] = widget

        # Random seed input field
        if not advancedOpts['randomSeed'][-1]:
            widget = widgets.IntText(
                value=advancedOpts['randomSeed'][0],
                description='Random seed:',
                style={'description_width': 'initial'},
                disabled=False
            )
            self._widgetsExtraParams['randomSeed'] = widget

        # @todo: this block of commented code can be used readily used to fix issue #95
        # if not advancedOpts['final_x'][-1]:
        #     opts = []
        #     for reactant in sorted(initialState.keys(), key=str):
        #         # opts.append( ( "Reactant " + r'$'+ latex(Symbol(str(reactant)))+r'$', str(reactant) ) )
        #         opts.append(("Reactant " + r'\(' + utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) + r'\)', str(reactant)))
        #     dropdown = widgets.Dropdown(
        #         options=opts,
        #         description='Final distribution (x axis):',
        #         value=advancedOpts['final_x'][0],
        #         style={'description_width': 'initial'}
        #     )
        #     self._widgetsPlotOnly['final_x'] = dropdown
        # if not advancedOpts['final_y'][-1]:
        #     opts = []
        #     for reactant in sorted(initialState.keys(), key=str):
        #           opts.append( ( r'$'+ utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) +'$' , str(reactant) ) )
        #           print("the reactant is " + str(reactant))
        #           print("the utils._greekPrependify(str(reactant) is " + str(utils._greekPrependify(str(reactant))) )
        #           print("the utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) is " + str(utils._doubleUnderscorify(utils._greekPrependify(str(reactant)))) )
        #         opts.append(("Reactant " + r'\(' + utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) + r'\)', str(reactant)))
        #     dropdown = widgets.Dropdown(
        #         options=opts,
        #         description='Final distribution (y axis):',
        #         value=advancedOpts['final_y'][0],
        #         style={'description_width': 'initial'}
        #     )
        #     self._widgetsPlotOnly['final_y'] = dropdown

        # @todo: this block of commented code can be used readily used to fix issue #283
        # # Checkbox for proportions or full populations plot
        # if not advancedOpts['plotProportions'][-1]:
        #     widget = widgets.Checkbox(
        #         value=advancedOpts['plotProportions'][0],
        #         description='Plot population proportions',
        #         disabled=False
        #     )
        #     self._widgetsPlotOnly['plotProportions'] = widget

        # Number of runs slider
        if not advancedOpts['runs'][-1]:
            runs = advancedOpts['runs']
            widget = widgets.IntSlider(value=runs[0], min=runs[1],
                                       max=runs[2], step=runs[3],
                                       readout_format='.' + str(utils._count_sig_decimals(str(runs[3]))) + 'f',
                                       description='Number of runs:',
                                       style={'description_width': 'initial'},
                                       # layout=widgets.Layout(width='50%'),
                                       disabled=False,
                                       continuous_update=continuousReplot)
            self._widgetsExtraParams['runs'] = widget

        # Checkbox for realtime plot update
        if not advancedOpts['aggregateResults'][-1]:
            widget = widgets.Checkbox(
                value=advancedOpts['aggregateResults'][0],
                description='Aggregate results',
                disabled=False
            )
            self._widgetsExtraParams['aggregateResults'] = widget

        return None

    def _orderAdvancedWidgets(self, _noInitialState):
        # define the widget order
        # self._extraWidgetsOrder.append('final_x')
        # self._extraWidgetsOrder.append('final_y')
        # self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('runs')
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('aggregateResults')


class MuMoTstochasticSimulationController(MuMoTcontroller):
    """Controller for stochastic simulations (base class of MuMoTmultiagentController)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._downloadWidget.on_click(self._download_link_unsupported, remove=True)
        self._downloadWidget.on_click(self._reveal_download_link)

    def _createAdvancedWidgets(self, SSParams, continuousReplot=False):
        initialState = SSParams['initialState'][0]
        if not SSParams['initialState'][-1]:
            # for state,pop in initialState.items():
            for i, state in enumerate(sorted(initialState.keys(), key=str)):
                pop = initialState[state]
                widget = widgets.FloatSlider(value=pop[0],
                                             min=pop[1],
                                             max=pop[2],
                                             step=pop[3],
                                             readout_format='.' + str(utils._count_sig_decimals(str(pop[3]))) + 'f',
                                             description=r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(Symbol('Phi_{' + str(state) + '}')))) + '$' + " at t=0: ",
                                             # description = r'\(' + latex(Symbol('Phi_'+str(state))) + r'\)' + " at t=0: ",
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                # disable one population widget (if there are more than 1) to constrain the initial sum of population sizes to 1
                if len(initialState) > 1:
                    if SSParams['substitutedReactant'][0] is None and i == 0:
                        # if there is not a 'substituted' reactant, the last population widget
                        widget.disabled = True
                    elif SSParams['substitutedReactant'][0] is not None and state == SSParams['substitutedReactant'][0]:
                        # if there is a 'substituted' reactant, this is the chosen one as the disabled pop
                        widget.disabled = True
                    else:
                        widget.observe(self._updateInitialStateWidgets, 'value')

                self._widgetsExtraParams[f"init{state}"] = widget

        # Max time slider
        if not SSParams['maxTime'][-1]:
            maxTime = SSParams['maxTime']
            widget = widgets.FloatSlider(value=maxTime[0], min=maxTime[1],
                                         max=maxTime[2], step=maxTime[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(maxTime[3]))) + 'f',
                                         description='Simulation time:',
                                         style={'description_width': 'initial'},
                                         # layout=widgets.Layout(width='50%'),
                                         disabled=False,
                                         continuous_update=continuousReplot)
            self._widgetsExtraParams['maxTime'] = widget

        # Random seed input field
        if not SSParams['randomSeed'][-1]:
            widget = widgets.IntText(
                value=SSParams['randomSeed'][0],
                description='Random seed:',
                style={'description_width': 'initial'},
                disabled=False
            )
            self._widgetsExtraParams['randomSeed'] = widget

        try:
            # Toggle buttons for plotting style
            if not SSParams['visualisationType'][-1]:
                plotToggle = widgets.ToggleButtons(
                    options=[('Temporal evolution', 'evo'), ('Final distribution', 'final'), ('Barplot', 'barplot')],
                    value=SSParams['visualisationType'][0],
                    description='Plot:',
                    disabled=False,
                    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                    tooltips=['Population change over time', 'Population distribution in each state at final timestep', 'Barplot of states at final timestep'],
                    # icons=['check'] * 3
                )
                plotToggle.observe(self._updateFinalViewWidgets, 'value')
                self._widgetsPlotOnly['visualisationType'] = plotToggle

        except widgets.trait_types.traitlets.TraitError:  # this widget could be redefined in a subclass and the init-value in SSParams['visualisationType'][0] might raise an exception
            pass

        if not SSParams['final_x'][-1] and (SSParams['visualisationType'][-1] is False or SSParams['visualisationType'][0] == 'final'):
            opts = []
            for reactant in sorted(initialState.keys(), key=str):
                # opts.append( ( "Reactant " + r'$'+ latex(Symbol(str(reactant)))+r'$', str(reactant) ) )
                opts.append(("Reactant " + r'\(' + utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) + r'\)', str(reactant)))
            dropdown = widgets.Dropdown(
                options=opts,
                description='Final distribution (x axis):',
                value=SSParams['final_x'][0],
                style={'description_width': 'initial'}
            )
            if SSParams['visualisationType'][0] != 'final':
                dropdown.layout.display = 'none'
            self._widgetsPlotOnly['final_x'] = dropdown
        if not SSParams['final_y'][-1] and (SSParams['visualisationType'][-1] is False or SSParams['visualisationType'][0] == 'final'):
            opts = []
            for reactant in sorted(initialState.keys(), key=str):
                # opts.append( ( r'$'+ utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) +'$' , str(reactant) ) )
                # print("the reactant is " + str(reactant))
                # print("the utils._greekPrependify(str(reactant) is " + str(utils._greekPrependify(str(reactant))) )
                # print("the utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) is " + str(utils._doubleUnderscorify(utils._greekPrependify(str(reactant)))) )
                opts.append(("Reactant " + r'\(' + utils._doubleUnderscorify(utils._greekPrependify(str(reactant))) + r'\)', str(reactant)))
            dropdown = widgets.Dropdown(
                options=opts,
                description='Final distribution (y axis):',
                value=SSParams['final_y'][0],
                style={'description_width': 'initial'}
            )
            if SSParams['visualisationType'][0] != 'final':
                dropdown.layout.display = 'none'
            self._widgetsPlotOnly['final_y'] = dropdown

        # Checkbox for proportions or full populations plot
        if not SSParams['plotProportions'][-1]:
            widget = widgets.Checkbox(
                value=SSParams['plotProportions'][0],
                description='Plot population proportions',
                disabled=False
            )
            self._widgetsPlotOnly['plotProportions'] = widget

        # Checkbox for realtime plot update
        if not SSParams['realtimePlot'][-1]:
            widget = widgets.Checkbox(
                value=SSParams['realtimePlot'][0],
                description='Runtime plot update',
                disabled=False
            )
            self._widgetsExtraParams['realtimePlot'] = widget

        # Number of runs slider
        if not SSParams['runs'][-1]:
            runs = SSParams['runs']
            widget = widgets.IntSlider(value=runs[0], min=runs[1],
                                       max=runs[2], step=runs[3],
                                       readout_format='.' + str(utils._count_sig_decimals(str(runs[3]))) + 'f',
                                       description='Number of runs:',
                                       style={'description_width': 'initial'},
                                       # layout=widgets.Layout(width='50%'),
                                       disabled=False,
                                       continuous_update=continuousReplot)
            self._widgetsExtraParams['runs'] = widget

        # Checkbox for realtime plot update
        if not SSParams['aggregateResults'][-1]:
            widget = widgets.Checkbox(
                value=SSParams['aggregateResults'][0],
                description='Aggregate results',
                disabled=False
            )
            self._widgetsPlotOnly['aggregateResults'] = widget

        self._addSpecificWidgets(SSParams, continuousReplot)

        return initialState

    def _addSpecificWidgets(self, SSParams, continuousReplot):
        pass

    def _orderAdvancedWidgets(self, initialState):
        # define the widget order
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append(f"init{state}")
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('visualisationType')
        self._extraWidgetsOrder.append('final_x')
        self._extraWidgetsOrder.append('final_y')
        self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('realtimePlot')
        self._extraWidgetsOrder.append('runs')
        self._extraWidgetsOrder.append('aggregateResults')


class MuMoTmultiagentController(MuMoTstochasticSimulationController):
    """Controller for multiagent views."""

    def _addSpecificWidgets(self, MAParams, continuousReplot=False):

        # Network type dropdown selector
        if not MAParams['netType'][-1]:
            netDropdown = widgets.Dropdown(
                options=[('Full graph', consts.NetworkType.FULLY_CONNECTED),
                         ('Erdos-Renyi', consts.NetworkType.ERSOS_RENYI),
                         ('Barabasi-Albert', consts.NetworkType.BARABASI_ALBERT),
                         # @todo: add network topology generated by random points in space
                         ('Moving particles', consts.NetworkType.DYNAMIC)
                         ],
                description='Network topology:',
                value=utils._decodeNetworkTypeFromString(MAParams['netType'][0]),
                style={'description_width': 'initial'},
                disabled=False
            )
            netDropdown.observe(self._update_net_params, 'value')
            self._widgetsExtraParams['netType'] = netDropdown

        # Network connectivity slider
        if not MAParams['netParam'][-1]:
            netParam = MAParams['netParam']
            widget = widgets.FloatSlider(value=netParam[0],
                                         min=netParam[1],
                                         max=netParam[2],
                                         step=netParam[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(netParam[3]))) + 'f',
                                         description='Network connectivity parameter',
                                         style={'description_width': 'initial'},
                                         layout=widgets.Layout(width='50%'),
                                         continuous_update=continuousReplot,
                                         disabled=False)
            self._widgetsExtraParams['netParam'] = widget

        # Agent speed
        if not MAParams['particleSpeed'][-1]:
            particleSpeed = MAParams['particleSpeed']
            widget = widgets.FloatSlider(value=particleSpeed[0],
                                         min=particleSpeed[1], max=particleSpeed[2], step=particleSpeed[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(particleSpeed[3]))) + 'f',
                                         description='Particle speed',
                                         style={'description_width': 'initial'},
                                         layout=widgets.Layout(width='50%'),
                                         continuous_update=continuousReplot,
                                         disabled=False)
            self._widgetsExtraParams['particleSpeed'] = widget

        # Random walk correlatedness
        if not MAParams['motionCorrelatedness'][-1]:
            motionCorrelatedness = MAParams['motionCorrelatedness']
            widget = widgets.FloatSlider(value=motionCorrelatedness[0],
                                         min=motionCorrelatedness[1],
                                         max=motionCorrelatedness[2],
                                         step=motionCorrelatedness[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(motionCorrelatedness[3]))) + 'f',
                                         description='Correlatedness of the random walk',
                                         layout=widgets.Layout(width='50%'),
                                         style={'description_width': 'initial'},
                                         continuous_update=continuousReplot,
                                         disabled=False)
            self._widgetsExtraParams['motionCorrelatedness'] = widget

        # Time scaling slider
        if not MAParams['timestepSize'][-1]:
            timestepSize = MAParams['timestepSize']
            widget = widgets.FloatSlider(value=timestepSize[0],
                                         min=timestepSize[1],
                                         max=timestepSize[2],
                                         step=timestepSize[3],
                                         readout_format='.' + str(utils._count_sig_decimals(str(timestepSize[3]))) + 'f',
                                         description='Timestep size',
                                         style={'description_width': 'initial'},
                                         layout=widgets.Layout(width='50%'),
                                         continuous_update=continuousReplot)
            self._widgetsExtraParams['timestepSize'] = widget

        # Toggle buttons for plotting style
        if not MAParams['visualisationType'][-1]:
            plotToggle = widgets.ToggleButtons(
                options=[('Temporal evolution', 'evo'), ('Network', 'graph'), ('Final distribution', 'final'), ('Barplot', 'barplot')],
                value=MAParams['visualisationType'][0],
                description='Plot:',
                disabled=False,
                button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                tooltips=['Population change over time', 'Population distribution in each state at final timestep', 'Barplot of states at final timestep'],
            )
            plotToggle.observe(self._updateFinalViewWidgets, 'value')
            self._widgetsPlotOnly['visualisationType'] = plotToggle

        # Particle display checkboxes
        if not MAParams['showTrace'][-1]:
            widget = widgets.Checkbox(
                value=MAParams['showTrace'][0],
                description='Show particle trace',
                disabled=False  # not (self._widgetsExtraParams['netType'].value == consts.NetworkType.DYNAMIC)
            )
            self._widgetsPlotOnly['showTrace'] = widget
        if not MAParams['showInteractions'][-1]:
            widget = widgets.Checkbox(
                value=MAParams['showInteractions'][0],
                description='Show communication links',
                disabled=False  # not (self._widgetsExtraParams['netType'].value == consts.NetworkType.DYNAMIC)
            )
            self._widgetsPlotOnly['showInteractions'] = widget

    def _orderAdvancedWidgets(self, initialState):
        # define the widget order
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append(f"init{state}")
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('timestepSize')
        self._extraWidgetsOrder.append('netType')
        self._extraWidgetsOrder.append('netParam')
        self._extraWidgetsOrder.append('particleSpeed')
        self._extraWidgetsOrder.append('motionCorrelatedness')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('visualisationType')
        self._extraWidgetsOrder.append('final_x')
        self._extraWidgetsOrder.append('final_y')
        self._extraWidgetsOrder.append('showTrace')
        self._extraWidgetsOrder.append('showInteractions')
        self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('realtimePlot')
        self._extraWidgetsOrder.append('runs')
        self._extraWidgetsOrder.append('aggregateResults')

    def _update_net_params(self, _=None):
        """Update the widgets related to the ``netType``

        It is linked - through ``observe()`` - before the ``_view`` is created.

        """
        if self._view:
            self._view._update_net_params(True)


class MuMoTmultiController(MuMoTcontroller):
    """Multi-view controller."""

    # replot function list to invoke on views
    _replotFunctions = None

    def __init__(self, controllers, params=None, initWidgets=None, **kwargs):
        if initWidgets is None:
            initWidgets = {}

        self._silent = kwargs.get('silent', False)
        self._replotFunctions = []
        fixedParamNames = None
        paramValuesDict = {}
        paramLabelDict = {}
        showPlotLimits = False
        showSystemSize = False
        views_ = []
        subPlotNum = 1
        model = None
        # @todo assuming same model for all views.
        # This operation is NOT correct when multicotroller views have different models
        # paramValuesDict = controllers[0]._view._mumotModel._create_free_param_dictionary_for_controller(inputParams=params if params is not None else [], initWidgets=initWidgets, showSystemSize=True, showPlotLimits=True )

        if params is not None:
            (fixedParamNames, fixedParamValues) = utils._process_params(params)
        for controller in controllers:
            # pass through the fixed params to each constituent view
            view = controller._view
            if params is not None:
                # view._fixedParams = dict(zip(fixedParamNames, fixedParamValues))
                # view._fixedParams = {**dict(zip(fixedParamNames, fixedParamValues)), **view._fixedParams}
                #     this operation merge the two dictionaries with the second overriding the values of the first

                # This operation merges the two dictionaries with the second overriding the values of the first.
                view._set_fixedParams({**dict(zip(fixedParamNames, fixedParamValues)), **view._fixedParams})
            for name, value in controller._widgetsFreeParams.items():
                # if params is None or name not in fixedParamNames:
                #    paramValueDict[name] = (value.value, value.min, value.max, value.step)
                if name in initWidgets:
                    paramValuesDict[name] = utils._parse_input_keyword_for_numeric_widgets(
                        inputValue=utils._get_item_from_params_list(params if params is not None else [], name),
                        defaultValueRangeStep=[defaults.MuMoTdefault._initialRateValue,
                                               defaults.MuMoTdefault._rateLimits[0],
                                               defaults.MuMoTdefault._rateLimits[1],
                                               defaults.MuMoTdefault._rateStep],
                        initValueRangeStep=initWidgets.get(name),
                        validRange=(-float("inf"), float("inf")))
                else:
                    paramValuesDict[name] = (value.value,
                                             value.min,
                                             value.max,
                                             value.step,
                                             not(params is None or name not in map(str, view._fixedParams.keys())))
            if controller._plotLimitsWidget is not None:
                showPlotLimits = True
                if 'plotLimits' in initWidgets:
                    paramValuesDict['plotLimits'] = utils._parse_input_keyword_for_numeric_widgets(
                        inputValue=utils._get_item_from_params_list(params if params is not None else [], 'plotLimits'),
                        defaultValueRangeStep=[defaults.MuMoTdefault._plotLimits,
                                               defaults.MuMoTdefault._plotLimitsLimits[0],
                                               defaults.MuMoTdefault._plotLimitsLimits[1],
                                               defaults.MuMoTdefault._plotLimitsStep],
                        initValueRangeStep=initWidgets.get('plotLimits'),
                        validRange=(-float("inf"), float("inf")))
                else:
                    paramValuesDict['plotLimits'] = (controller._plotLimitsWidget.value,
                                                     controller._plotLimitsWidget.min,
                                                     controller._plotLimitsWidget.max,
                                                     controller._plotLimitsWidget.step,
                                                     not(params is None or 'plotLimits' not in map(str, view._fixedParams.keys())))
            if controller._systemSizeWidget is not None:
                showSystemSize = True
                if 'systemSize' in initWidgets:
                    paramValuesDict['systemSize'] = utils._parse_input_keyword_for_numeric_widgets(
                        inputValue=utils._get_item_from_params_list(params if params is not None else [], 'systemSize'),
                        defaultValueRangeStep=[defaults.MuMoTdefault._systemSize,
                                               defaults.MuMoTdefault._systemSizeLimits[0],
                                               defaults.MuMoTdefault._systemSizeLimits[1],
                                               defaults.MuMoTdefault._systemSizeStep],
                        initValueRangeStep=initWidgets.get('systemSize'),
                        validRange=(1, float("inf")))
                else:
                    paramValuesDict['systemSize'] = (controller._systemSizeWidget.value,
                                                     controller._systemSizeWidget.min,
                                                     controller._systemSizeWidget.max,
                                                     controller._systemSizeWidget.step,
                                                     not(params is None or 'systemSize' not in map(str, view._fixedParams.keys())))
            paramLabelDict.update(controller._paramLabelDict)
            # for name, value in controller._widgetsExtraParams.items():
            # widgetsExtraParamsTmp[name] = value
            if type(controller) is MuMoTmultiController:
                # if controller._replotFunction is None: # presume this controller is a multi controller (@todo check?)
                for view in controller._view._views:
                    views_.append(view)

                # if view._controller._replotFunction is None: # presume this controller is a multi controller (@todo check?)
                for func, _, axes3d in controller._replotFunctions:
                    self._replotFunctions.append((func, subPlotNum, axes3d))
                # else:
                #    self._replotFunctions.append((view._controller._replotFunction, subPlotNum, view._axes3d))
            else:
                views_.append(controller._view)
                # if controller._replotFunction is None: # presume this controller is a multi controller (@todo check?)
                #     for func, foo in controller._replotFunctions:
                #         self._replotFunctions.append((func, subPlotNum))
                # else:
                self._replotFunctions.append((controller._replotFunction,
                                              subPlotNum,
                                              controller._view._axes3d))
            subPlotNum += 1
            # check if all views refer to same model
            if model is None:
                model = view._mumotModel
            elif model != view._mumotModel:
                raise exceptions.MuMoTValueError(
                    'Multicontroller views do not all refer to same model')

        # for view in self._views:
        #     # presume this controller is a multi controller (@todo check?)
        #     if view._controller._replotFunction is None:
        #         for func in view._controller._replotFunctions:
        #             self._replotFunctions.append(func)
        #     else:
        #         self._replotFunctions.append(view._controller._replotFunction)
        #     view._controller = self
        super().__init__(paramValuesDict, paramLabelDict, False,
                         showPlotLimits, showSystemSize, params=params,
                         **kwargs)

        # handle Extra and PlotOnly params
        addProgressBar = False
        for controller in controllers:
            # retrieve the _widgetsExtraParams from each controller
            for name, widget in controller._widgetsExtraParams.items():
                widget.unobserve(controller._replotFunction, 'value')
                self._widgetsExtraParams[name] = widget
            # retrieve the _widgetsPlotOnly from each controller
            for name, widget in controller._widgetsPlotOnly.items():
                widget.unobserve(controller._redrawFunction, 'value')
                # in multiController, plotProportions=True is forced
                if name == "plotProportions":
                    widget.value = True
                    widget.disabled = True
                # this is necessary due to limit visualisation-type to only 'evo' and 'final'
                if name == "visualisationType":
                    # Toggle buttons for plotting style
                    widget = widgets.ToggleButtons(
                        options=[('Temporal evolution', 'evo'), ('Final distribution', 'final')],
                        value=widget.value,
                        description='Plot:',
                        disabled=False,
                        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                        tooltips=['Population change over time', 'Population distribution in each state at final timestep'],
                    )
                    widget.observe(self._updateFinalViewWidgets, 'value')
                self._widgetsPlotOnly[name] = widget
            # retrieve the _extraWidgetsOrder from each controller
            self._extraWidgetsOrder.extend(x for x in controller._extraWidgetsOrder
                                           if x not in self._extraWidgetsOrder)
            # if controller._progressBar:
            #     addProgressBar = True
        if self._widgetsExtraParams or self._widgetsPlotOnly:
            # set widgets to possible initial/fixed values if specified in the multi-controller
            # for key, value in kwargs.items():
            for key in kwargs.keys() | initWidgets.keys():
                inputValue = kwargs.get(key)
                ep1 = None
                ep2 = None
                if key == 'xlab':
                    for controller in controllers:
                        controller._view._xlab = kwargs.get('xlab')
                if key == 'ylab':
                    for controller in controllers:
                        controller._view._ylab = kwargs.get('ylab')
                if key == 'fontsize':
                    for controller in controllers:
                        controller._view._axes_font_size = kwargs.get('fontsize')
                if key == 'legend_loc':
                    for controller in controllers:
                        controller._view._legend_loc = kwargs.get('legend_loc')
                if key == 'legend_fontsize':
                    for controller in controllers:
                        controller._view._legend_fontsize = kwargs.get('legend_fontsize')
                if key == 'choose_yrange':
                    for controller in controllers:
                        controller._view._chooseYrange = kwargs.get('choose_yrange')
                if key == 'choose_xrange':
                    for controller in controllers:
                        controller._view._chooseXrange = kwargs.get('choose_xrange')
                if key == 'initialState':
                    ep1 = views_[0]._mumotModel._getAllReactants()
                    ep2 = [True, [react for react in views_[0]._mumotModel._getAllReactants()[0] if react not in views_[0]._mumotModel._reactants][0] if views_[0]._mumotModel._systemSize is not None else None]
                    # @todo assuming same model for all views.
                    # This operation is NOT correct when multicotroller views have different models.
                if key == 'visualisationType':
                    ep1 = "multicontroller"
                if key == 'final_x' or key == 'final_y':
                    ep1 = views_[0]._mumotModel._getAllReactants()[0]
                    # @todo assuming same model for all views.
                    # This operation is NOT correct when multicotroller views have different models.
                if key == 'netParam':
                    ep1 = [kwargs.get('netType', self._widgetsExtraParams.get('netType')),
                           kwargs.get('netType') is not None]
                    maxSysSize = 1
                    for view in views_:
                        maxSysSize = max(maxSysSize, view._getSystemSize())
                    ep2 = maxSysSize
                optionValues = utils._format_advanced_option(
                    optionName=key,
                    inputValue=inputValue,
                    initValues=initWidgets.get(key),
                    extraParam=ep1,
                    extraParam2=ep2)
                # if option is fixed
                if optionValues[-1] is True:
                    if key == 'initialState':  # initialState is special
                        for state, pop in optionValues[0].items():
                            optionValues[0][state] = pop[0]
                            stateKey = "init" + str(state)
                            # delete the widgets
                            if stateKey in self._widgetsExtraParams:
                                del self._widgetsExtraParams[stateKey]
                    if key == 'netType':  # netType is special
                        optionValues[0] = utils._decodeNetworkTypeFromString(optionValues[0])  # @todo: if only netType (and not netParam) is specified, then multicotroller won't work...
                    if key == 'visualisationType' and optionValues[0] == 'final':  # visualisationType == 'final' is special
                        if self._widgetsPlotOnly.get('final_x') is not None:
                            self._widgetsPlotOnly['final_x'].layout.display = 'flex'
                        if self._widgetsPlotOnly.get('final_y') is not None:
                            self._widgetsPlotOnly['final_y'].layout.display = 'flex'
                    # set the value in all the views
                    for view in views_:
                        view._fixedParams[key] = optionValues[0]
                    # delete the widgets
                    if key in self._widgetsExtraParams:
                        del self._widgetsExtraParams[key]
                    if key in self._widgetsPlotOnly:
                        del self._widgetsPlotOnly[key]
                else:
                    # update the values with the init values
                    if key in self._widgetsExtraParams:
                        if len(optionValues) == 5:
                            self._widgetsExtraParams[key].max = 10**7  # temp to avoid exception min>max
                            self._widgetsExtraParams[key].min = optionValues[1]
                            self._widgetsExtraParams[key].max = optionValues[2]
                            self._widgetsExtraParams[key].step = optionValues[3]
                            self._widgetsExtraParams[key].readout_format = '.' + str(utils._count_sig_decimals(str(optionValues[3]))) + 'f'
                        self._widgetsExtraParams[key].value = optionValues[0]
                    if key in self._widgetsPlotOnly:
                        if len(optionValues) == 5:
                            self._widgetsPlotOnly[key].max = 10**7  # temp to avoid exception min>max
                            self._widgetsPlotOnly[key].min = optionValues[1]
                            self._widgetsPlotOnly[key].max = optionValues[2]
                            self._widgetsPlotOnly[key].step = optionValues[3]
                            self._widgetsPlotOnly[key].readout_format = '.' + str(utils._count_sig_decimals(str(optionValues[3]))) + 'f'
                        self._widgetsPlotOnly[key].value = optionValues[0]
                    if key == 'initialState':
                        for state, pop in optionValues[0].items():
                            # self._widgetsExtraParams['init'+str(state)].unobserve(self._updateInitialStateWidgets, 'value')
                            self._widgetsExtraParams[f"init{state}"].max = float('inf')  # temp to avoid exception min>max
                            self._widgetsExtraParams[f"init{state}"].min = pop[1]
                            self._widgetsExtraParams[f"init{state}"].max = pop[2]
                            self._widgetsExtraParams[f"init{state}"].step = pop[3]
                            self._widgetsExtraParams[f"init{state}"].readout_format = '.' + str(utils._count_sig_decimals(str(pop[3]))) + 'f'
                            self._widgetsExtraParams[f"init{state}"].value = pop[0]
                            # self._widgetsExtraParams['init'+str(state)].observe(self._updateInitialStateWidgets, 'value')

            # create the "Advanced options" tab
            if not self._silent:
                self._displayAdvancedOptionsTab()

        # if necessary adding the progress bar
        if addProgressBar:
            # Loading bar (useful to give user progress status for long executions)
            self._progressBar = widgets.FloatProgress(
                value=0,
                min=0,
                max=(self._widgetsExtraParams['maxTime'].value
                     if self._widgetsExtraParams.get('maxTime') is not None
                     else views_[0]._fixedParams.get('maxTime')),
                # step=1,
                description='Loading:',
                bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
                style={'description_width': 'initial'},
                orientation='horizontal'
            )
            if not self._silent:
                display(self._progressBar)

        self._view = views.MuMoTmultiView(self, model, views_, controllers, subPlotNum - 1, **kwargs)
        if fixedParamNames is not None:
            # self._view._fixedParams = dict(zip(fixedParamNames, fixedParamValues))
            self._view._set_fixedParams(dict(zip(fixedParamNames, fixedParamValues)))

        for controller in controllers:
            controller._setErrorWidget(self._errorMessage)

        # @todo handle correctly the re-draw only widgets and function
        self._setReplotFunction(self._view._plot, self._view._plot)

        # silent = kwargs.get('silent', False)
        if not self._silent:
            self._view._plot()
