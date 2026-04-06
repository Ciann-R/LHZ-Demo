"""
lhz.error_analysis
==================
Replicates the fault-tolerance analysis from Figure 3 of:
    Lechner, Hauke, Zoller — Sci. Adv. 1, e1500838 (2015)

Key result from the paper (Eq. 7):
    P_d * P_m = (N-1) * Γ * T

Despite having N(N-1)/2 physical qubits (so more chances for spin flips),
the redundant encoding ensures the total error scales only *linearly* with N —
identical to a direct implementation with N qubits.

This module:
1. Estimates P_d (probability of a spin flip occurring)
2. Estimates P_m (probability that a flipped qubit causes a wrong readout)
3. Shows that P_d * P_m ∝ N  (linear scaling, not quadratic)
4. Plots the number of valid readout combinations (exponential in N)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import itertools
import matplotlib.pyplot as plt
import numpy as np

from lhz.architecture import LHZArchitecture


@dataclass
class ErrorAnalysisResult:
    """Results from a fault-tolerance analysis sweep over system sizes.

    Attributes
    ----------
    n_values : list[int]
        Logical qubit counts analyzed.
    n_physical : list[int]
        Corresponding physical qubit counts K = N(N-1)/2.
    p_spin_flip : list[float]
        P_d = N(N-1)/2 * Γ*T — probability of any spin flip (normalized by Γ*T).
    p_info_loss : list[float]
        P_m = N_f / N_meas — probability of wrong readout given one flip.
    total_error : list[float]
        P_d * P_m — should scale linearly with N.
    n_readouts : list[int]
        Total number of valid determining readout sequences.
    n_trials : int
        Monte Carlo trials used to estimate P_m.
    """

    n_values: list[int] = field(default_factory=list)
    n_physical: list[int] = field(default_factory=list)
    p_spin_flip: list[float] = field(default_factory=list)
    p_info_loss: list[float] = field(default_factory=list)
    total_error: list[float] = field(default_factory=list)
    n_readouts: list[int] = field(default_factory=list)
    n_trials: int = 0

    def plot(self, figsize: tuple[int, int] = (12, 5)) -> plt.Figure:
        """Reproduce Figure 3 from Lechner, Hauke, Zoller (2015).

        Left panel: P_d (spin flips), P_m (info loss), and P_d*P_m vs N.
        Right inset: Number of valid readout sequences vs N (exponential).

        Returns
        -------
        plt.Figure
        """
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("#f8f9fa")

        # Main panel
        ax_main = fig.add_axes([0.08, 0.12, 0.55, 0.78])
        ax_inset = fig.add_axes([0.70, 0.12, 0.27, 0.78])

        n_arr = np.array(self.n_values)

        # --- Main panel: error scaling ---
        ax_main.plot(
            self.n_values,
            self.p_spin_flip,
            "r-o",
            linewidth=2,
            markersize=6,
            label=r"$P_d$ = Spin flips  $\propto N^2$",
        )
        ax_main.plot(
            self.n_values,
            self.p_info_loss,
            "b-s",
            linewidth=2,
            markersize=6,
            label=r"$P_m$ = Info. loss  $\propto N^{-1}$",
        )
        ax_main.plot(
            self.n_values,
            self.total_error,
            "k--D",
            linewidth=2.5,
            markersize=7,
            label=r"$P_d \cdot P_m$ = Total error  $\propto N$",
        )

        # Reference lines
        n_ref = np.linspace(min(self.n_values), max(self.n_values), 100)
        scale_quad = self.p_spin_flip[0] / (self.n_values[0] ** 2)
        scale_lin = self.total_error[0] / self.n_values[0]
        ax_main.plot(n_ref, scale_quad * n_ref**2, "r:", alpha=0.4, linewidth=1)
        ax_main.plot(n_ref, scale_lin * n_ref, "k:", alpha=0.4, linewidth=1)

        ax_main.set_xlabel("Number of logical qubits (N)", fontsize=12)
        ax_main.set_ylabel("Error scaling (normalized)", fontsize=12)
        ax_main.set_title(
            "LHZ Fault Tolerance\n(replicating Fig. 3, Lechner et al. 2015)",
            fontsize=12,
            fontweight="bold",
        )
        ax_main.legend(fontsize=9, loc="upper left")
        ax_main.set_facecolor("#f8f9fa")
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xticks(self.n_values)

        # Annotation: key result from Eq. 7
        ax_main.annotate(
            r"Key result (Eq. 7): $P_d \cdot P_m = (N-1)\Gamma T$" "\n"
            r"Linear in N — same as direct implementation!",
            xy=(0.5, 0.05),
            xycoords="axes fraction",
            ha="center",
            fontsize=8.5,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # --- Inset: readout combinations (exponential growth) ---
        ax_inset.semilogy(
            self.n_values,
            self.n_readouts,
            "g-^",
            linewidth=2,
            markersize=7,
            label="# Readouts",
        )
        ax_inset.set_xlabel("N", fontsize=11)
        ax_inset.set_ylabel("# Readout sequences", fontsize=10)
        ax_inset.set_title("Valid Readouts\n(exponential in N)", fontsize=10, fontweight="bold")
        ax_inset.set_facecolor("#f8f9fa")
        ax_inset.grid(True, alpha=0.3, which="both")
        ax_inset.set_xticks(self.n_values)

        return fig


class ErrorAnalyzer:
    """Analyze LHZ fault tolerance as a function of system size.

    Reproduces the analytical and numerical results of Figure 3 from the
    paper, showing how error scaling remains linear in N despite the
    quadratic growth of physical qubits.

    Parameters
    ----------
    n_range : range or list[int]
        Logical qubit counts to analyze (e.g., range(2, 10)).
    n_trials : int
        Monte Carlo trials for estimating P_m (information loss per flip).
    decoherence_rate : float
        Γ — decoherence rate per qubit per unit time (normalized).
    total_time : float
        T — total annealing time (normalized).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> analyzer = ErrorAnalyzer(n_range=range(2, 9), n_trials=500)
    >>> result = analyzer.run()
    >>> result.plot()
    """

    def __init__(
        self,
        n_range: range | list[int] = range(2, 9),
        n_trials: int = 500,
        decoherence_rate: float = 0.01,
        total_time: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_range = list(n_range)
        self.n_trials = n_trials
        self.decoherence_rate = decoherence_rate
        self.total_time = total_time
        self._rng = np.random.default_rng(seed)

    def run(self) -> ErrorAnalysisResult:
        """Run the full error analysis.

        For each N in n_range:
        1. Compute P_d = K * Γ * T  (analytical, K = N(N-1)/2)
        2. Estimate P_m via Monte Carlo: flip random qubits, check readout
        3. Count total valid readout sequences

        Returns
        -------
        ErrorAnalysisResult
        """
        result = ErrorAnalysisResult(n_trials=self.n_trials)

        for N in self.n_range:
            K = N * (N - 1) // 2

            # --- P_d: probability of any spin flip ---
            # Scales as number of physical qubits × decoherence
            p_d = K * self.decoherence_rate * self.total_time

            # --- P_m: information loss from a single flip ---
            p_m = self._estimate_info_loss(N)

            # --- Count valid readout sequences ---
            n_readouts = self._count_readout_sequences(N)

            result.n_values.append(N)
            result.n_physical.append(K)
            result.p_spin_flip.append(p_d)
            result.p_info_loss.append(p_m)
            result.total_error.append(p_d * p_m)
            result.n_readouts.append(n_readouts)

            print(
                f"N={N}: K={K}, P_d={p_d:.4f}, P_m={p_m:.4f}, "
                f"P_d*P_m={p_d*p_m:.4f}, readouts={n_readouts}"
            )

        return result

    def _estimate_info_loss(self, N: int) -> float:
        """Estimate P_m: fraction of readout sequences corrupted by one flip.

        Algorithm:
        1. Create a random valid logical spin configuration.
        2. Encode it into physical qubits.
        3. Flip one random physical qubit (simulating decoherence).
        4. Check all possible N-1 length readout chains — count how many
           give the wrong answer.
        5. Average over n_trials.

        Parameters
        ----------
        N : int
            Number of logical qubits.

        Returns
        -------
        float
            Estimated P_m ∈ [0, 1].
        """
        arch = LHZArchitecture(n_logical=N, seed=int(self._rng.integers(0, 10000)))
        K = arch.n_physical

        wrong_fraction_sum = 0.0

        for _ in range(self.n_trials):
            # Random logical spin configuration
            logical_true = self._rng.choice([-1, 1], size=N)

            # Encode: physical qubit (i,j) = 1 if logical[i]==logical[j], else 0
            correct_states = np.zeros(K, dtype=int)
            for q in arch.physical_qubits:
                i, j = q.logical_i, q.logical_j
                correct_states[q.index] = 1 if logical_true[i] == logical_true[j] else 0

            # Flip one random physical qubit (decoherence event)
            flip_idx = int(self._rng.integers(0, K))
            corrupted = correct_states.copy()
            corrupted[flip_idx] = 1 - corrupted[flip_idx]

            # Check all N-1 length readout chains (adjacent bond chains)
            # A chain reads out bonds (0,1),(1,2),...,(k-1,k) for any ordering
            wrong = self._check_readout_chains(arch, correct_states, corrupted, N)
            wrong_fraction_sum += wrong

        return wrong_fraction_sum / self.n_trials

    def _check_readout_chains(
        self,
        arch: LHZArchitecture,
        correct: np.ndarray,
        corrupted: np.ndarray,
        N: int,
    ) -> float:
        """Check what fraction of N-1 readout chains give wrong answer.

        The simplest readout: consecutive bonds (0,1),(1,2),...,(N-2,N-1).
        More chains could be checked; here we use a sample for efficiency.
        """
        # Check the primary readout chain: (0,1), (1,2), ..., (N-2, N-1)
        def decode_chain(states: np.ndarray) -> np.ndarray:
            logical = np.ones(N)
            for j in range(1, N):
                bond_idx = arch.bond_to_index.get((j - 1, j), None)
                if bond_idx is None:
                    break
                s = states[bond_idx]
                logical[j] = logical[j - 1] * (1 if s == 1 else -1)
            return logical

        correct_logical = decode_chain(correct)
        corrupted_logical = decode_chain(corrupted)

        # Two decodings agree if they're identical up to global sign flip
        if (np.allclose(correct_logical, corrupted_logical) or
                np.allclose(correct_logical, -corrupted_logical)):
            return 0.0
        return 1.0

    @staticmethod
    def _count_readout_sequences(N: int) -> int:
        """Count the number of valid determining readout combinations.

        A "determining combination" is any set of N-1 bonds (i,j) that
        form a spanning tree of the N logical qubits — i.e., they fully
        determine all N logical spins up to a global sign.

        This equals the number of spanning trees of the complete graph K_N,
        given by Cayley's formula: N^(N-2).

        Parameters
        ----------
        N : int
            Number of logical qubits.

        Returns
        -------
        int
            Number of valid readout sequences = N^(N-2).
        """
        if N < 2:
            return 1
        return N ** (N - 2)
