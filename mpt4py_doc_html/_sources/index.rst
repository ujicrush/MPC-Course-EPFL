.. mpt4py documentation master file, created by
   sphinx-quickstart on Mon Feb 24 22:05:18 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mpt4py
======

About
------

The **Multi-Parametric Toolbox for Python** (``mpt4py``) is an open-source, Python-based toolbox for 
parametric optimization, computational geometry and model predictive control, 
covering the core functionalities of the `MPT3 <https://www.mpt3.org>`_ in Matlab.


Credits
-------

The ``mpt4py`` is developed at the `Laboratoire d'Automatique <https://www.epfl.ch/labs/la/>`_ at `École Polytechnique Fédérale de Lausanne (EPFL) <https://www.epfl.ch/>`_, Switzerland. This work was supported by the `Swiss National Science Foundation <https://www.snf.ch/>`_ under the `NCCR Automation <https://nccr-automation.ch/>`_ (grant agreement 51NF40_180545).

The ``mpt4py`` is built upon the following open-source packages:

* `CVXPY <https://www.cvxpy.org>`_: Interface for optimization problems and convex set representation.
* `pycddlib <https://github.com/mcmtroffaes/pycddlib>`_: Used to perform polyhedral operations like vertex/facet enumeration, redundancy elimination, and projection.
* `PyVista <https://docs.pyvista.org>`_: Optional backend for visualization of geometry objects.
* `python-control <https://python-control.readthedocs.io/en/0.10.1/>`_: Used to model dynamical systems.

.. toctree::
   :maxdepth: 4
   :caption: Navigation:
   :hidden:

   .. installation.rst
   contents.rst
   .. tutorials.md
   

.. raw:: html

   <br><br>

.. image:: figures/epfl_logo.png
  :width: 150
  :alt: EPFL

.. image:: figures/nccr_automation_logo.svg
  :width: 200
  :alt: NCCR-Automation


FAQ
-------

- What are the differences between `mpt4py` and other polyhedron computation libraries like `pympt` or `pycddlib`?

   - `mpt4py` is a high-level library that provides a user-friendly interface for polyhedral computations, while `pympt` and `pycddlib` are lower-level libraries that focus on specific algorithms and data structures. `mpt4py` is built on top of these libraries to provide a more convenient and efficient way to work with polyhedra in Python.