"""Numerical continuation of equilibria, used to draw bifurcation diagrams.

MuMoT only talks to the continuation code through this module, so the
numerical backend can be swapped without touching the views.  The interface
is expressed purely in terms of SymPy symbols/expressions and NumPy arrays:

* :class:`EquilibriumContinuation` -- continues equilibria of the ODE system
  ``dx/dt = f(x; p)`` in one free parameter, either from a given state
  (:meth:`~EquilibriumContinuation.from_state`) or by switching branch at a
  previously detected branch point
  (:meth:`~EquilibriumContinuation.from_branch_point`).
* :class:`Branch` -- the result: parameter values, state values and Jacobian
  eigenvalues along the curve plus the special points detected on it.
* :class:`SpecialPoint` -- a limit point (fold, ``'LP'``) or branch point
  (``'BP'``).

The current backend wraps the copy of PyDSTool's PyCont that is bundled in
:mod:`mumot._vendor.pydstool`.  A replacement backend (e.g. a native
pseudo-arclength continuation on top of NumPy/SciPy) needs to subclass
:class:`EquilibriumContinuation`, implement the two ``from_*`` methods
returning :class:`Branch` objects, and be returned by
:func:`equilibrium_continuation`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import sympy

__all__ = ['Branch', 'EquilibriumContinuation', 'SpecialPoint',
           'equilibrium_continuation']


@dataclass
class SpecialPoint:
    """A special point detected on a :class:`Branch`."""
    #: ``'LP'`` (limit point / fold) or ``'BP'`` (branch point)
    kind: str
    #: 1-based index of this point among the points of the same kind on its branch
    index: int
    #: value of the free parameter at the point
    parameter: float
    #: values of the state variables at the point
    state: Dict[sympy.Symbol, float]
    #: backend-specific data needed to switch branch at this point
    _backend_data: Any = field(default=None, repr=False, compare=False)


@dataclass
class Branch:
    """A curve of equilibria computed by an :class:`EquilibriumContinuation`."""
    #: values of the free parameter along the branch, shape ``(n,)``
    parameter: np.ndarray
    #: values of each state variable along the branch, each of shape ``(n,)``
    states: Dict[sympy.Symbol, np.ndarray]
    #: eigenvalues of the Jacobian along the branch, shape ``(n, dim)``
    eigenvalues: np.ndarray
    #: special points (limit points and branch points) found on the branch
    special_points: List[SpecialPoint]
    #: directions (``'backward'``/``'forward'``) in which continuation failed
    failures: List[str] = field(default_factory=list)

    def special(self, kind: str) -> List[SpecialPoint]:
        """Special points of the given kind (``'LP'`` or ``'BP'``), in order."""
        return [sp for sp in self.special_points if sp.kind == kind]


class EquilibriumContinuation:
    """Continuation of the equilibria of ``dx/dt = f(x; p)`` in one parameter.

    Parameters
    ----------
    equations
        Right-hand side of the ODE for each state variable.
    parameters
        Values of all the other symbols appearing in ``equations``, including
        ``free_parameter`` (whose value is the default starting value).
    free_parameter
        The parameter varied along the branches.
    max_num_points, max_step_size, min_step_size
        Continuation settings.
    """

    def __init__(self, equations: Mapping[sympy.Symbol, sympy.Expr],
                 parameters: Mapping[sympy.Symbol, float],
                 free_parameter: sympy.Symbol, *, max_num_points: int = 100,
                 max_step_size: float = 1e-1, min_step_size: float = 1e-5):
        self.equations = dict(equations)
        self.parameters = dict(parameters)
        self.free_parameter = free_parameter
        self.max_num_points = max_num_points
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        missing = (set().union(*(sympy.sympify(eq).free_symbols for eq in self.equations.values()))
                   - set(self.equations) - set(self.parameters) - {free_parameter})
        if missing:
            raise ValueError(f"No values given for {sorted(map(str, missing))}")

    def from_state(self, state: Mapping[sympy.Symbol, float],
                   parameter_value: Optional[float] = None,
                   step_size: float = 2e-3) -> Optional[Branch]:
        """Continue the equilibrium branch through ``state`` in both directions.

        Returns ``None`` if no branch could be computed.
        """
        raise NotImplementedError

    def from_branch_point(self, point: SpecialPoint,
                          step_size: float = 5e-3) -> Optional[Branch]:
        """Continue the other branch through the branch point ``point``.

        Returns ``None`` if no branch could be computed.
        """
        raise NotImplementedError


def equilibrium_continuation(*args, **kwargs) -> EquilibriumContinuation:
    """Create an :class:`EquilibriumContinuation` using the default backend.

    Takes the same arguments as :class:`EquilibriumContinuation`.
    """
    return _PyDSToolEquilibriumContinuation(*args, **kwargs)


class _PyDSToolEquilibriumContinuation(EquilibriumContinuation):
    """Backend using PyCont from the bundled copy of PyDSTool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PyDSTool restricts the names it accepts (and sorts state variables
        # by name), so give every symbol an opaque name, ordering the state
        # variables as MuMoT has always presented them to PyDSTool.
        self._var_names = {sym: f"X{i:03d}" for i, sym in
                           enumerate(sorted(self.equations, key=_legacy_pydstool_name))}
        par_syms = sorted(set(self.parameters) | {self.free_parameter}, key=str)
        self._par_names = {sym: f"P{i:03d}" for i, sym in enumerate(par_syms)}
        to_names = {sym: sympy.Symbol(name) for sym, name in
                    {**self._var_names, **self._par_names}.items()}
        self._varspecs = {self._var_names[sym]: str(sympy.sympify(eq).xreplace(to_names))
                          for sym, eq in self.equations.items()}
        self._sym_of_var = {name: sym for sym, name in self._var_names.items()}
        self._free_name = self._par_names[self.free_parameter]
        self._branch_count = 0

    # -- PyDSTool plumbing ------------------------------------------------
    def _new_continuer(self, ics, parameter_value):
        from ._vendor import pydstool as dst
        model = dst.args(name=f"MuMoT_model_{id(self)}")
        model.varspecs = dict(self._varspecs)
        model.pars = {self._par_names[sym]: float(value)
                      for sym, value in self.parameters.items()}
        ode = dst.Generator.Vode_ODEsystem(model)
        ode.set(ics=ics)
        if parameter_value is not None:
            ode.set(pars={self._free_name: float(parameter_value)})
        return dst.ContClass(ode)

    def _continue(self, cont, step_size, **extra_args):
        from ._vendor import pydstool as dst
        self._branch_count += 1
        name = f"B{self._branch_count}"
        cont_args = dst.args(name=name, type='EP-C')
        cont_args.freepars = [self._free_name]
        cont_args.MaxNumPoints = self.max_num_points
        cont_args.MaxStepSize = self.max_step_size
        cont_args.MinStepSize = self.min_step_size
        cont_args.StepSize = step_size
        cont_args.LocBifPoints = ['LP', 'BP']
        cont_args.SaveEigen = True
        for key, value in extra_args.items():
            setattr(cont_args, key, value)
        cont.newCurve(cont_args)

        failures = []
        for direction in ('backward', 'forward'):
            try:
                getattr(cont[name], direction)()
            except Exception:
                failures.append(direction)
        return self._to_branch(cont, name, failures)

    def _to_branch(self, cont, name, failures):
        curve = cont[name]
        sol = curve.sol
        if sol is None:
            return None
        states = {sym: np.asarray(sol[var]) for var, sym in self._sym_of_var.items()}
        eigenvalues = np.array([sol[k].labels['EP']['data'].evals for k in range(len(sol))])
        special_points = []
        for kind in ('LP', 'BP'):
            index = 1
            while True:
                point = curve.getSpecialPoint(kind + str(index))
                if not point:
                    break
                special_points.append(SpecialPoint(
                    kind=kind, index=index,
                    parameter=point[self._free_name],
                    state={sym: point[var] for var, sym in self._sym_of_var.items()},
                    _backend_data=(cont, name, point)))
                index += 1
        return Branch(parameter=np.asarray(sol[self._free_name]), states=states,
                      eigenvalues=eigenvalues, special_points=special_points,
                      failures=failures)

    # -- public interface -------------------------------------------------
    def from_state(self, state, parameter_value=None, step_size=2e-3):
        ics = {self._var_names[sym]: value for sym, value in state.items()
               if sym in self._var_names}
        cont = self._new_continuer(ics, parameter_value)
        return self._continue(cont, step_size)

    def from_branch_point(self, point, step_size=5e-3):
        if point.kind != 'BP':
            raise ValueError("Can only switch branch at a branch point ('BP')")
        cont, curve_name, pyds_point = point._backend_data
        extra = {'initpoint': f"{curve_name}:BP{point.index}"}
        # Start along the direction of the new branch computed when the branch
        # point was found; otherwise PyCont estimates a tangent at the
        # (singular) branch point, and which branch it follows is then down to
        # floating-point rounding.
        bp_data = pyds_point.labels['BP']['data']
        if getattr(bp_data, 'branch', None) is not None:
            extra['initdirec'] = dict(bp_data.branch)
        return self._continue(cont, step_size, **extra)


def _legacy_pydstool_name(symbol) -> str:
    """Name MuMoT historically gave ``symbol`` when passing it to PyDSTool.

    Only used to keep the order in which state variables are presented to
    PyDSTool unchanged.
    """
    name = str(symbol)
    for c in ('{', '}', '_', '\\', '^'):
        name = name.replace(c, '')
    if name[0].islower() or name in ('gamma', 'Gamma'):
        name = 'A' + name
    return name
