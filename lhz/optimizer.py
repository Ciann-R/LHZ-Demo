"""
lhz.optimizer
=============
Encode combinatorial optimization problems into the LHZ architecture
and solve them via simulated annealing on the physical qubit lattice.

Supported problems
------------------
- Max-Cut  : Partition graph vertices to maximize cut edges.
- TSP      : Travelling Salesman Problem (small instances, QUBO encoding).

The key LHZ insight used here: J_ij become local fields on physical qubits,
so the annealing only touches local degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from lhz.architecture import LHZArchitecture


@dataclass
class OptimizationResult:
    """Results from an LHZ optimization run.

    Attributes
    ----------
    problem : str
        Name of the optimization problem solved.
    best_energy : float
        Lowest total energy found during the anneal.
    best_states : np.ndarray
        Physical qubit states achieving best_energy.
    best_logical : np.ndarray
        Decoded logical spin configuration (±1 values).
    energy_history : list[float]
        Energy at each accepted step (for convergence plots).
    cut_value : int or None
        For Max-Cut problems: number of edges crossing the partition.
    cut_partition : tuple[set, set] or None
        For Max-Cut: the two vertex sets.
    n_sweeps : int
        Number of Monte Carlo sweeps performed.
    acceptance_rate : float
        Fraction of proposed spin flips that were accepted.
    """

    problem: str
    best_energy: float
    best_states: np.ndarray
    best_logical: np.ndarray
    energy_history: list[float] = field(default_factory=list)
    cut_value: Optional[int] = None
    cut_partition: Optional[tuple[set, set]] = None
    n_sweeps: int = 0
    acceptance_rate: float = 0.0

    def visualize_solution(
        self,
        graph: Optional[nx.Graph] = None,
        figsize: tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """Plot the optimization result: energy convergence and (optionally) graph solution.

        Parameters
        ----------
        graph : nx.Graph, optional
            If provided and problem is Max-Cut, draw the graph with the partition.
        figsize : tuple[int, int]
            Figure size.

        Returns
        -------
        plt.Figure
        """
        n_panels = 2 if (graph is not None and self.cut_partition is not None) else 1
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        # Panel 1: Energy convergence
        ax = axes[0]
        ax.plot(self.energy_history, color="#e74c3c", linewidth=1.0, alpha=0.8)
        ax.axhline(
            self.best_energy,
            color="#2ecc71",
            linewidth=2,
            linestyle="--",
            label=f"Best energy: {self.best_energy:.4f}",
        )
        ax.set_xlabel("Accepted step", fontsize=11)
        ax.set_ylabel("Energy", fontsize=11)
        ax.set_title("Simulated Annealing Convergence", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_facecolor("#f8f9fa")
        ax.grid(True, alpha=0.3)

        # Panel 2: Graph solution (Max-Cut)
        if n_panels == 2 and graph is not None and self.cut_partition is not None:
            ax2 = axes[1]
            set_a, set_b = self.cut_partition
            pos = nx.spring_layout(graph, seed=42)
            color_map = ["#3498db" if v in set_a else "#e74c3c" for v in graph.nodes()]

            # Draw cut / non-cut edges differently
            cut_edges = [
                (u, v)
                for u, v in graph.edges()
                if (u in set_a) != (v in set_a)
            ]
            non_cut_edges = [e for e in graph.edges() if e not in cut_edges]

            nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=500, ax=ax2)
            nx.draw_networkx_labels(graph, pos, ax=ax2, font_color="white", font_weight="bold")
            nx.draw_networkx_edges(
                graph, pos, edgelist=non_cut_edges, ax=ax2, alpha=0.3, width=1.5
            )
            nx.draw_networkx_edges(
                graph, pos, edgelist=cut_edges, ax=ax2,
                edge_color="#2ecc71", width=3.0, style="dashed", alpha=0.9
            )

            ax2.set_title(
                f"Max-Cut Solution\nCut value: {self.cut_value} edges",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_facecolor("#f8f9fa")
            ax2.axis("off")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#3498db", label="Partition A"),
                Patch(facecolor="#e74c3c", label="Partition B"),
            ]
            ax2.legend(handles=legend_elements, loc="upper left", fontsize=9)

        fig.suptitle(
            f"LHZ Optimization — {self.problem}  "
            f"(acceptance rate: {self.acceptance_rate:.1%})",
            fontsize=11,
            y=1.01,
        )
        plt.tight_layout()
        return fig


class LHZOptimizer:
    """Solve combinatorial optimization problems using the LHZ architecture.

    Encodes the problem as an Ising Hamiltonian, builds the LHZ physical
    qubit lattice, then runs simulated annealing on the *physical* qubits
    with local field updates — exactly the programmable LHZ protocol.

    Parameters
    ----------
    problem : nx.Graph or np.ndarray
        - nx.Graph → solved as Max-Cut (edge weights used as couplings)
        - np.ndarray (N×N) → treated as a raw Ising coupling matrix
    constraint_strength : float
        Relative strength of plaquette constraints. Paper suggests C/J ≈ 2.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.petersen_graph()
    >>> opt = LHZOptimizer(G)
    >>> result = opt.solve(n_sweeps=2000, T_start=5.0, T_end=0.01)
    >>> print(f"Cut: {result.cut_value}")
    """

    def __init__(
        self,
        problem: nx.Graph | np.ndarray,
        constraint_strength: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.constraint_strength = constraint_strength
        self._graph: Optional[nx.Graph] = None

        if isinstance(problem, nx.Graph):
            self._graph = problem
            coupling_matrix = self._maxcut_couplings(problem)
            self._problem_name = "Max-Cut"
        elif isinstance(problem, np.ndarray):
            coupling_matrix = problem
            self._problem_name = "Ising"
        else:
            raise TypeError("problem must be a networkx Graph or numpy ndarray.")

        n_logical = coupling_matrix.shape[0]
        self.arch = LHZArchitecture(
            n_logical=n_logical,
            coupling_matrix=coupling_matrix,
            constraint_strength=constraint_strength,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Problem encoders
    # ------------------------------------------------------------------

    @staticmethod
    def _maxcut_couplings(G: nx.Graph) -> np.ndarray:
        """Convert a Max-Cut graph to an Ising coupling matrix.

        Max-Cut objective: maximize Σ_{(i,j)∈E} (1 - s_i * s_j) / 2
        Equivalent Ising: minimize Σ_{(i,j)∈E} J_ij * s_i * s_j
        with J_ij = -w_ij (negative edge weight → antiparallel spins preferred).

        Parameters
        ----------
        G : nx.Graph
            Graph with optional 'weight' edge attributes.

        Returns
        -------
        np.ndarray
            N×N Ising coupling matrix.
        """
        nodes = sorted(G.nodes())
        n = len(nodes)
        node_idx = {v: i for i, v in enumerate(nodes)}
        J = np.zeros((n, n))
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            i, j = node_idx[u], node_idx[v]
            J[i, j] = -w  # negative → antiparallel preferred
            J[j, i] = -w
        return J

    # ------------------------------------------------------------------
    # Annealing
    # ------------------------------------------------------------------

    def solve(
        self,
        n_sweeps: int = 1000,
        T_start: float = 5.0,
        T_end: float = 0.01,
        schedule: str = "geometric",
    ) -> OptimizationResult:
        """Run simulated annealing on the LHZ physical qubit lattice.

        At each step, a random physical qubit is flipped and the move is
        accepted/rejected via the Metropolis criterion. Temperature decreases
        from T_start to T_end over n_sweeps sweeps.

        Parameters
        ----------
        n_sweeps : int
            Number of Monte Carlo sweeps (each sweep = n_physical flip attempts).
        T_start : float
            Initial temperature (high → accepts bad moves freely).
        T_end : float
            Final temperature (low → essentially greedy).
        schedule : str
            Cooling schedule: "geometric" (exponential) or "linear".

        Returns
        -------
        OptimizationResult
        """
        arch = self.arch
        n_physical = arch.n_physical
        arch.randomize_states()

        # Temperature schedule
        if schedule == "geometric":
            temperatures = np.geomspace(T_start, T_end, n_sweeps)
        else:
            temperatures = np.linspace(T_start, T_end, n_sweeps)

        best_energy = arch.total_energy()
        best_states = arch.states.copy()
        energy_history: list[float] = [best_energy]
        n_accepted = 0
        n_total = 0

        for sweep_idx, T in enumerate(temperatures):
            for _ in range(n_physical):
                # Pick a random physical qubit to flip
                qubit_idx = int(self._rng.integers(0, n_physical))
                qubit = arch.physical_qubits[qubit_idx]

                if qubit.is_fixed:
                    continue

                # Compute energy change from flipping this qubit
                old_state = qubit.state
                old_energy = arch.total_energy()

                qubit.state = 1 - old_state  # flip
                new_energy = arch.total_energy()
                delta_E = new_energy - old_energy

                # Metropolis acceptance
                if delta_E < 0 or self._rng.random() < np.exp(-delta_E / T):
                    n_accepted += 1
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_states = arch.states.copy()
                else:
                    qubit.state = old_state  # reject: revert

                n_total += 1

            energy_history.append(arch.total_energy())

        # Decode solution
        arch.set_states(best_states)
        logical_spins = arch.decode_logical_spins()

        # Compute Max-Cut value if applicable
        cut_value = None
        cut_partition = None
        if self._graph is not None:
            cut_value, cut_partition = self._evaluate_maxcut(logical_spins)

        return OptimizationResult(
            problem=self._problem_name,
            best_energy=best_energy,
            best_states=best_states,
            best_logical=logical_spins,
            energy_history=energy_history,
            cut_value=cut_value,
            cut_partition=cut_partition,
            n_sweeps=n_sweeps,
            acceptance_rate=n_accepted / max(n_total, 1),
        )

    def _evaluate_maxcut(
        self, logical_spins: np.ndarray
    ) -> tuple[int, tuple[set, set]]:
        """Evaluate cut value for the decoded logical spin configuration."""
        nodes = sorted(self._graph.nodes())
        set_a = {nodes[i] for i, s in enumerate(logical_spins) if s > 0}
        set_b = {nodes[i] for i, s in enumerate(logical_spins) if s <= 0}

        cut_value = sum(
            1
            for u, v in self._graph.edges()
            if (u in set_a) != (v in set_b)  # crosses partition
        )
        return cut_value, (set_a, set_b)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Optional[nx.Graph]:
        """The graph being optimized (Max-Cut problems only)."""
        return self._graph