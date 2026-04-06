"""
tests/test_optimizer.py
=======================
Unit tests for lhz.optimizer.
"""

import networkx as nx
import numpy as np
import pytest

from lhz.optimizer import LHZOptimizer, OptimizationResult


class TestLHZOptimizerInit:
    """Test optimizer initialization from different problem types."""

    def test_from_graph(self):
        """Optimizer accepts a networkx Graph."""
        G = nx.cycle_graph(4)
        opt = LHZOptimizer(G, seed=0)
        assert opt.arch.n_logical == 4

    def test_from_matrix(self):
        """Optimizer accepts a raw coupling matrix."""
        J = np.random.default_rng(0).uniform(-1, 1, (4, 4))
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        opt = LHZOptimizer(J, seed=0)
        assert opt.arch.n_logical == 4

    def test_invalid_input_type(self):
        """Non-graph, non-array input should raise TypeError."""
        with pytest.raises(TypeError):
            LHZOptimizer("not a graph")


class TestMaxCutCouplings:
    """Test the Max-Cut → Ising encoding."""

    def test_negative_couplings_for_unit_weights(self):
        """Unit-weight Max-Cut should give J_ij = -1 for connected pairs."""
        G = nx.path_graph(3)  # 0-1-2
        opt = LHZOptimizer(G, seed=0)
        J = opt.arch.coupling_matrix
        # Nodes 0-1 and 1-2 are connected → J should be -1
        assert np.isclose(J[0, 1], -1.0)
        assert np.isclose(J[1, 2], -1.0)

    def test_disconnected_nodes_zero_coupling(self):
        """Disconnected node pairs should have J_ij = 0."""
        G = nx.path_graph(3)  # 0-1-2 (0 and 2 not directly connected)
        opt = LHZOptimizer(G, seed=0)
        J = opt.arch.coupling_matrix
        assert np.isclose(J[0, 2], 0.0)


class TestSolveResult:
    """Test that solve() returns a well-formed OptimizationResult."""

    def setup_method(self):
        G = nx.cycle_graph(4)
        self.opt = LHZOptimizer(G, seed=42)
        self.result = self.opt.solve(n_sweeps=100, T_start=2.0, T_end=0.1)

    def test_result_type(self):
        assert isinstance(self.result, OptimizationResult)

    def test_cut_value_nonnegative(self):
        assert self.result.cut_value >= 0

    def test_cut_value_bounded_by_edges(self):
        G = nx.cycle_graph(4)
        assert self.result.cut_value <= G.number_of_edges()

    def test_partition_covers_all_nodes(self):
        """Partition A ∪ B should equal all nodes."""
        G = nx.cycle_graph(4)
        a, b = self.result.cut_partition
        assert a | b == set(G.nodes())
        assert a & b == set()

    def test_energy_history_nonempty(self):
        assert len(self.result.energy_history) > 0

    def test_best_states_shape(self):
        assert self.result.best_states.shape == (self.opt.arch.n_physical,)

    def test_acceptance_rate_in_range(self):
        assert 0.0 <= self.result.acceptance_rate <= 1.0

    def test_logical_spins_pm1(self):
        """All decoded logical spins should be ±1."""
        spins = self.result.best_logical
        assert all(s in [1.0, -1.0] for s in spins)


class TestCycleGraphKnownSolution:
    """Cycle graph C4 has a known optimal Max-Cut of 4 (all edges cut)."""

    def test_c4_optimal_cut(self):
        """C4 has an optimal cut of 4 — alternating partitions."""
        G = nx.cycle_graph(4)
        opt = LHZOptimizer(G, seed=0)
        result = opt.solve(n_sweeps=3000, T_start=3.0, T_end=0.001)
        # The optimal cut for C4 is 4 edges
        assert result.cut_value >= 3  # allow some tolerance for small anneals