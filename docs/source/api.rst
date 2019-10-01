API
===

.. todo:: Add preamble before API documentation.  Note that primarily of interest to developers.

.. currentmodule:: mumot


Functions
---------

.. autosummary::
   :toctree: autosummary

   about
   parseModel
   setVerboseExceptions

Model classes
-------------

.. autosummary::
   :toctree: autosummary

   MuMoTmodel

View classes
------------

.. inheritance-diagram:: mumot.views
   :parts: 1

.. autosummary::
   :toctree: autosummary

   MuMoTview
   MuMoTmultiView
   MuMoTtimeEvolutionView
   MuMoTintegrateView
   MuMoTnoiseCorrelationsView
   MuMoTfieldView
   MuMoTvectorView
   MuMoTstreamView
   MuMoTbifurcationView
   MuMoTstochasticSimulationView
   MuMoTmultiagentView
   MuMoTSSAView

Controller classes
------------------

.. inheritance-diagram:: mumot.controllers
   :parts: 1

.. autosummary::
   :toctree: autosummary

   MuMoTcontroller
   MuMoTbifurcationController
   MuMoTtimeEvolutionController
   MuMoTstochasticSimulationController
   MuMoTmultiagentController
   MuMoTmultiController

Exception classes
-----------------

.. inheritance-diagram:: mumot.exceptions
   :parts: 1

.. autosummary::
   :toctree: autosummary

   MuMoTError
   MuMoTValueError
   MuMoTSyntaxError
