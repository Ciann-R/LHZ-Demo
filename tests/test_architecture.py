"""
tests/test_architecture.py
==========================
Unit tests for lhz.architecture — the core LHZ encoding.
"""

import numpy as np
import pytest

from lhz.architecture import LHZArchitecture, PhysicalQubit, Plaquette


class TestLHZScaling:
    """Test that qubit counts match the paper's formulas."""

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
    def test_n_physical_scaling(self, N):
        """Physical qubits K = N(N-1)/2."""
        arch = LHZArchitecture(n_logical=N, seed=0)
        assert arch.n_physical == N * (N - 1) // 2

    @pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
    def test_n_physical_qubits_created(self, N):
        """Exactly K physical qubit objects are created."""
        arch = LHZArchitecture(n_logical=N, seed=0)
        assert len(arch.physical_qubits) == arch.n_physical

    def test_invalid_n_raises(self):
        """n_logical < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            LHZArchitecture(n_logical=1)

    def test_wrong_coupling_matrix_shape(self):
        """Mismatched coupling matrix should raise ValueError."""
        bad_matrix = np.zeros((3, 3))
        with pytest.raises(ValueError):
            LHZArchitecture(n_logical=4, coupling_matrix=bad_matrix)


class TestBondMapping:
    """Test that logical bond → physical qubit mapping is correct."""

    def setup_method(self):
        self.arch = LHZArchitecture(n_logical=4, seed=42)

    def test_all_bonds_present(self):
        """Every (i,j) bond with i < j should have a physical qubit."""
        N = self.arch.n_logical
        for i in range(N):
            for j in range(i + 1, N):
                assert (i, j) in self.arch.bond_to_index

    def test_no_duplicate_indices(self):
        """Each physical qubit index is used exactly once."""
        indices = list(self.arch.bond_to_index.values())
        assert len(indices) == len(set(indices))

    def test_coupling_values_match_matrix(self):
        """Physical qubit couplings match the coupling matrix J_ij."""
        arch = self.arch
        for q in arch.physical_qubits:
            expected = arch.coupling_matrix[q.logical_i, q.logical_j]
            assert np.isclose(q.coupling, expected)


class TestStateManagement:
    """Test qubit state manipulation and consistency."""

    def setup_method(self):
        self.arch = LHZArchitecture(n_logical=4, seed=0)

    def test_states_shape(self):
        """States array has shape (K,)."""
        assert self.arch.states.shape == (self.arch.n_physical,)

    def test_set_states_roundtrip(self):
        """set_states / states roundtrip is lossless."""
        new_states = np.array([1, 0, 1, 1, 0, 1])
        self.arch.set_states(new_states)
        assert np.array_equal(self.arch.states, new_states)

    def test_set_states_wrong_length(self):
        """set_states with wrong length should raise ValueError."""
        with pytest.raises(ValueError):
            self.arch.set_states(np.array([1, 0, 1]))

    def test_states_binary(self):
        """All states should be 0 or 1."""
        self.arch.randomize_states(seed=7)
        states = self.arch.states
        assert set(states).issubset({0, 1})


class TestDecoding:
    """Test logical spin decoding from physical qubit states."""

    def test_all_parallel_decodes_to_all_same(self):
        """If all physical qubits = 1 (parallel), all logical spins match."""
        arch = LHZArchitecture(n_logical=4, seed=0)
        arch.set_states(np.ones(arch.n_physical, dtype=int))
        logical = arch.decode_logical_spins()
        # All spins should be +1 (or all -1 — gauge freedom)
        assert np.all(logical == logical[0])

    def test_alternating_antiparallel(self):
        """Chain of antiparallel bonds → alternating logical spins."""
        arch = LHZArchitecture(n_logical=4, seed=0)
        # Set (0,1),(1,2),(2,3) all to 0 (antiparallel)
        states = np.ones(arch.n_physical, dtype=int)
        for j in range(1, 4):
            states[arch.bond_to_index[(j - 1, j)]] = 0
        arch.set_states(states)
        logical = arch.decode_logical_spins()
        # Should alternate: +1, -1, +1, -1
        expected_pattern = np.array([1.0, -1.0, 1.0, -1.0])
        assert (np.allclose(logical, expected_pattern) or
                np.allclose(logical, -expected_pattern))

    def test_decode_length(self):
        """Decoded logical spin array has length N."""
        for N in [3, 4, 5]:
            arch = LHZArchitecture(n_logical=N, seed=0)
            logical = arch.decode_logical_spins()
            assert len(logical) == N


class TestEnergy:
    """Test energy calculations."""

    def test_energy_is_finite(self):
        """Energy should always be a finite float."""
        arch = LHZArchitecture(n_logical=4, seed=1)
        assert np.isfinite(arch.total_energy())

    def test_zero_coupling_energy(self):
        """With all J_ij = 0, local field energy is zero."""
        J = np.zeros((4, 4))
        arch = LHZArchitecture(n_logical=4, coupling_matrix=J)
        assert np.isclose(arch.local_field_energy(), 0.0)

    def test_constraint_energy_satisfied(self):
        """If all plaquettes satisfied, constraint energy should be minimal."""
        arch = LHZArchitecture(n_logical=3, seed=0)
        # All-1 state satisfies all parity constraints (0 antialigned = even)
        arch.set_states(np.ones(arch.n_physical, dtype=int))
        assert arch.count_violated_constraints() == 0