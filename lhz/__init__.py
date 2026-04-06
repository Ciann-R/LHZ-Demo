"""
LHZ Quantum Toolkit
===================
A simulator for the Lechner-Hauke-Zoller (LHZ) parity quantum annealing architecture.

Based on: Lechner, Hauke, Zoller — Science Advances, 2015.
DOI: 10.1126/sciadv.1500838

Modules
-------
architecture : Core LHZ encoding, physical qubit layout, constraint lattice
optimizer    : Problem encoder and simulated annealer (Max-Cut, TSP)
error_analysis : Fault tolerance analysis vs. system size
"""

from lhz.architecture import LHZArchitecture
from lhz.optimizer import LHZOptimizer, OptimizationResult
from lhz.error_analysis import ErrorAnalyzer
__version__ = "0.1.0"
__all__ = ["LHZArchitecture", "LHZOptimizer", "OptimizationResult", "ErrorAnalyzer"]