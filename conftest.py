"""pytest configuration shared by the unit tests and the nbval notebook runs."""

try:
    from nbval.kernel import RunningKernel
except ImportError:  # nbval not installed: only the unit tests are run
    RunningKernel = None


if RunningKernel is not None:
    def _execute_cell_input(self, cell_input, allow_stdin=None):
        """Execute a notebook cell, recording it in the kernel's input history.

        nbval (>= 0.10) executes cells with ``store_history=False``, unlike
        Jupyter itself. MuMoT's documented way of defining a model is a
        ``%%model`` cell followed by ``mumot.parseModel(In[n])``, which reads
        that history, so run cells as Jupyter does.
        """
        return self.kc.execute(
            cell_input,
            store_history=True,
            allow_stdin=allow_stdin,
            stop_on_error=False,
        )

    RunningKernel.execute_cell_input = _execute_cell_input
