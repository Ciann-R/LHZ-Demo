"""
lhz.architecture
================
Core implementation of the LHZ parity encoding.

Maps N logical qubits (all-to-all Ising model) to K = N(N-1)/2 physical
qubits arranged on a 2D lattice with local 4-body plaquette constraints.

Reference: Lechner, Hauke, Zoller — Sci. Adv. 1, e1500838 (2015)
           Equations (1)–(4) and Figure 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PhysicalQubit:
    """A single physical qubit in the LHZ lattice.

    Each physical qubit represents the *relative alignment* of two logical
    qubits i and j:
        state = 1  →  logical qubits i, j are parallel   (same spin)
        state = 0  →  logical qubits i, j are antiparallel (opposite spin)

    Attributes
    ----------
    index : int
        Flat index in the physical qubit array.
    logical_i : int
        First logical qubit this bond connects.
    logical_j : int
        Second logical qubit this bond connects.
    coupling : float
        J_ij value from the original Ising problem. Becomes a local field
        on this physical qubit (key LHZ insight).
    state : int
        Current spin state: 1 (parallel) or 0 (antiparallel).
    is_fixed : bool
        True for auxiliary qubits fixed to 1 (boundary row, see Fig. 1D).
    row : int
        Row position in the triangular lattice layout.
    col : int
        Column position in the triangular lattice layout.
    """

    index: int
    logical_i: int
    logical_j: int
    coupling: float = 0.0
    state: int = 1
    is_fixed: bool = False
    row: int = 0
    col: int = 0

    def __repr__(self) -> str:
        return (
            f"PhysicalQubit(idx={self.index}, bond=({self.logical_i},{self.logical_j}), "
            f"J={self.coupling:.3f}, state={self.state})"
        )


@dataclass
class Plaquette:
    """A 4-body parity constraint on a square plaquette of physical qubits.

    The constraint enforces that the number of 0's (antiparallel bonds)
    among the four physical qubits must be even (0, 2, or 4). This is
    derived from closed-loop consistency of the logical spin configuration.

    Equivalently (Eq. 4, four-body form):
        C_l = -C * s̃_n * s̃_e * s̃_s * s̃_w  (in ±1 convention)

    Attributes
    ----------
    qubit_indices : list[int]
        Indices of the four physical qubits forming this plaquette,
        ordered [north, east, south, west].
    position : tuple[int, int]
        (row, col) of this plaquette in the lattice.
    """

    qubit_indices: list[int]
    position: tuple[int, int] = (0, 0)

    def is_satisfied(self, states: np.ndarray) -> bool:
        """Check if this plaquette constraint is satisfied.

        Parameters
        ----------
        states : np.ndarray
            Array of all physical qubit states (0 or 1).

        Returns
        -------
        bool
            True if the number of 0-states among the four qubits is even.
        """
        vals = states[self.qubit_indices]
        n_zeros = int(np.sum(vals == 0))
        return n_zeros % 2 == 0


def _lhz_plaquette_bonds(
    n: int,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Return interior and boundary plaquette bond lists for N logical qubits.

    Separates the two cases the paper distinguishes (Fig. 1D):
      - Interior: 4-bond square plaquettes tiling the bulk of the lattice.
      - Boundary: 3-bond triangles along the bottom readout row.

    Each plaquette is a list of (i, j) bond tuples representing logical qubit
    pairs. Caller is responsible for translating bonds to physical qubit indices.

    Parameters
    ----------
    n : int
        Number of logical qubits N.

    Returns
    -------
    interior : list of 4-element bond lists
    boundary : list of 3-element bond lists
    """
    interior = []
    for s in range(1, n - 2):
        for i in range(0, n - s - 2):
            interior.append([
                (i,     i + s + 1),
                (i + 1, i + s + 1),
                (i + 1, i + s + 2),
                (i,     i + s + 2),
            ])

    boundary = []
    for i in range(0, n - 2):
        boundary.append([
            (i,     i + 1),
            (i,     i + 2),
            (i + 1, i + 2),
        ])

    return interior, boundary


class LHZArchitecture:
    """LHZ parity encoding architecture.

    Encodes an N-logical-qubit all-to-all Ising problem into K = N(N-1)/2
    physical qubits on a triangular lattice with local plaquette constraints.

    Parameters
    ----------
    n_logical : int
        Number of logical qubits N. Must be >= 2.
    coupling_matrix : np.ndarray, optional
        N×N symmetric matrix of Ising couplings J_ij (upper triangle used).
        If None, a random matrix is generated for demonstration.
    constraint_strength : float
        Strength C of the plaquette constraints relative to couplings.
        The paper recommends C/J ≈ 2 for reliable operation.
    seed : int, optional
        Random seed for reproducible coupling generation.

    Attributes
    ----------
    n_logical : int
    n_physical : int
        K = N(N-1)/2 — one physical qubit per logical bond.
    n_constraints : int
        K - N + 1 plaquette constraints needed to fix the gauge freedom.
    physical_qubits : list[PhysicalQubit]
    plaquettes : list[Plaquette]
    bond_to_index : dict[tuple[int,int], int]
        Maps (i, j) logical bond to physical qubit index.

    Examples
    --------
    >>> arch = LHZArchitecture(n_logical=4)
    >>> print(arch)
    LHZArchitecture(N=4 logical, K=6 physical, 3 constraints)
    >>> arch.visualize()
    """

    def __init__(
        self,
        n_logical: int,
        coupling_matrix: Optional[np.ndarray] = None,
        constraint_strength: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_logical < 2:
            raise ValueError("n_logical must be >= 2.")

        self.n_logical = n_logical
        self.constraint_strength = constraint_strength
        self._rng = np.random.default_rng(seed)

        # Build or validate coupling matrix
        if coupling_matrix is not None:
            if coupling_matrix.shape != (n_logical, n_logical):
                raise ValueError(
                    f"coupling_matrix must be {n_logical}×{n_logical}, "
                    f"got {coupling_matrix.shape}."
                )
            self.coupling_matrix = coupling_matrix.astype(float)
        else:
            self.coupling_matrix = self._random_couplings()

        # Derived counts
        self.n_physical = n_logical * (n_logical - 1) // 2
        self.n_constraints = self.n_physical - n_logical + 1

        # Build the architecture
        self.physical_qubits: list[PhysicalQubit] = []
        self.plaquettes: list[Plaquette] = []
        self.bond_to_index: dict[tuple[int, int], int] = {}

        self._build_physical_qubits()
        self._build_plaquettes()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _random_couplings(self) -> np.ndarray:
        """Generate a random symmetric Ising coupling matrix."""
        J = self._rng.uniform(-1.0, 1.0, (self.n_logical, self.n_logical))
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        return J

    def _build_physical_qubits(self) -> None:
        """Create one physical qubit per logical bond (i, j) with i < j.

        The triangular layout mirrors Fig. 1D of the paper: bond (i,j)
        is placed at row = j - i - 1, col = i.
        """
        idx = 0
        for i in range(self.n_logical):
            for j in range(i + 1, self.n_logical):
                row = j - i - 1
                col = i
                qubit = PhysicalQubit(
                    index=idx,
                    logical_i=i,
                    logical_j=j,
                    coupling=self.coupling_matrix[i, j],
                    state=self._rng.integers(0, 2),
                    row=row,
                    col=col,
                )
                self.physical_qubits.append(qubit)
                self.bond_to_index[(i, j)] = idx
                idx += 1

    def _build_plaquettes(self) -> None:
        """Build the local plaquette constraints using the LHZ geometry.

        Delegates geometry to _lhz_plaquette_bonds(), which explicitly
        separates interior 4-bond squares from boundary 3-bond triangles,
        matching the paper's Fig. 1D construction exactly.

        Interior plaquettes enforce even parity (0, 2, or 4 antiparallel bonds).
        Boundary triangles are the bottom readout row of the lattice.
        """
        interior, boundary = _lhz_plaquette_bonds(self.n_logical)

        for pos, bonds in enumerate(interior):
            indices = [self.bond_to_index[b] for b in bonds]
            self.plaquettes.append(
                Plaquette(qubit_indices=indices, position=(0, pos))
            )

        for pos, bonds in enumerate(boundary):
            indices = [self.bond_to_index[b] for b in bonds]
            self.plaquettes.append(
                Plaquette(qubit_indices=indices, position=(1, pos))
            )

    # ------------------------------------------------------------------
    # State manipulation
    # ------------------------------------------------------------------

    @property
    def states(self) -> np.ndarray:
        """Current spin states of all physical qubits as a numpy array."""
        return np.array([q.state for q in self.physical_qubits], dtype=int)

    def set_states(self, states: np.ndarray) -> None:
        """Set all physical qubit states.

        Parameters
        ----------
        states : np.ndarray
            Array of 0/1 values, length = n_physical.
        """
        if len(states) != self.n_physical:
            raise ValueError(f"Expected {self.n_physical} states, got {len(states)}.")
        for qubit, s in zip(self.physical_qubits, states):
            qubit.state = int(s)

    def randomize_states(self, seed: Optional[int] = None) -> None:
        """Randomly initialize all physical qubit states."""
        rng = np.random.default_rng(seed)
        for qubit in self.physical_qubits:
            if not qubit.is_fixed:
                qubit.state = int(rng.integers(0, 2))

    def decode_logical_spins(self) -> np.ndarray:
        """Decode physical qubit states back to logical spin configuration.

        Reads out the chain of bonds (0,1), (1,2), (2,3), ... which fully
        determines all logical spins up to a global inversion (see Fig. 1D).

        Returns
        -------
        np.ndarray
            Logical spin values in {+1, −1}, length = n_logical.
        """
        logical = np.ones(self.n_logical, dtype=float)
        # s[0] = +1 by convention (global gauge freedom)
        for j in range(1, self.n_logical):
            bond_idx = self.bond_to_index[(j - 1, j)]
            bond_state = self.physical_qubits[bond_idx].state
            # state=1 → parallel → same sign; state=0 → antiparallel → flip
            logical[j] = logical[j - 1] * (1 if bond_state == 1 else -1)
        return logical

    def count_violated_constraints(self) -> int:
        """Count the number of unsatisfied plaquette constraints."""
        states = self.states
        return sum(1 for p in self.plaquettes if not p.is_satisfied(states))

    def local_field_energy(self) -> float:
        """Compute energy from local fields (the J_k s̃_k terms in Eq. 3).

        In the LHZ encoding, J_ij becomes a local field on the physical
        qubit representing bond (i,j). We use the ±1 convention.
        """
        energy = 0.0
        for qubit in self.physical_qubits:
            spin = 2 * qubit.state - 1  # map {0,1} → {-1,+1}
            energy += qubit.coupling * spin
        return energy

    def constraint_energy(self) -> float:
        """Compute energy penalty from violated plaquette constraints."""
        penalty = 0.0
        states = self.states
        for plaquette in self.plaquettes:
            vals = states[plaquette.qubit_indices]
            # Four-body term: product of ±1 spins should be +1 for even parity
            spins = 2 * vals - 1
            product = float(np.prod(spins))
            penalty += -self.constraint_strength * product
        return penalty

    def total_energy(self) -> float:
        """Total Hamiltonian energy: local fields + constraint penalties."""
        return self.local_field_energy() + self.constraint_energy()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        title: str = "LHZ Parity Architecture",
        show_constraints: bool = True,
        figsize: tuple[int, int] = (10, 7),
    ) -> plt.Figure:
        """Visualize the physical qubit lattice and plaquette constraints.

        Produces a plot similar to Fig. 1D from the paper, showing:
        - Physical qubits as circles (colored by bond type)
        - Labels showing the logical bond (i,j)
        - Plaquette constraint regions highlighted in grey

        Parameters
        ----------
        title : str
            Plot title.
        show_constraints : bool
            Whether to draw plaquette constraint regions.
        figsize : tuple[int, int]
            Figure size in inches.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        ax.axis("off")

        spacing = 1.5
        radius = 0.35

        # Collect qubit positions
        positions: dict[int, tuple[float, float]] = {}
        for qubit in self.physical_qubits:
            x = qubit.col * spacing + qubit.row * spacing * 0.5
            y = -qubit.row * spacing
            positions[qubit.index] = (x, y)

        # Draw plaquette constraint regions
        if show_constraints:
            for plaquette in self.plaquettes:
                pts = [positions[idx] for idx in plaquette.qubit_indices if idx in positions]
                if len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    # Draw a shaded polygon
                    poly = mpatches.Polygon(
                        list(zip(xs, ys)),
                        closed=True,
                        alpha=0.12,
                        facecolor="#4a90d9",
                        edgecolor="#4a90d9",
                        linewidth=1.5,
                        linestyle="--",
                    )
                    ax.add_patch(poly)

        # Color map: bond index distance
        cmap = plt.cm.get_cmap("tab10")

        # Draw qubits
        for qubit in self.physical_qubits:
            x, y = positions[qubit.index]
            dist = qubit.logical_j - qubit.logical_i
            color = cmap(dist - 1)

            # Qubit circle
            circle = plt.Circle(
                (x, y),
                radius,
                color=color,
                ec="white",
                linewidth=2,
                zorder=3,
            )
            ax.add_patch(circle)

            # State indicator (inner dot)
            state_color = "white" if qubit.state == 1 else "#333333"
            inner = plt.Circle((x, y), radius * 0.45, color=state_color, zorder=4)
            ax.add_patch(inner)

            # Bond label
            ax.text(
                x,
                y,
                f"{qubit.logical_i},{qubit.logical_j}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="#333" if qubit.state == 1 else "white",
                zorder=5,
            )

            # Coupling value below
            ax.text(
                x,
                y - radius - 0.15,
                f"J={qubit.coupling:.2f}",
                ha="center",
                va="top",
                fontsize=6,
                color="#666",
                zorder=5,
            )

        # Legend
        legend_handles = []
        for d in range(1, self.n_logical):
            patch = mpatches.Patch(
                color=cmap(d - 1), label=f"Bond distance {d}"
            )
            legend_handles.append(patch)
        if show_constraints:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="#4a90d9",
                    alpha=0.3,
                    label="Plaquette constraint",
                    linestyle="--",
                    edgecolor="#4a90d9",
                )
            )

        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

        # Stats box
        stats = (
            f"N={self.n_logical} logical qubits\n"
            f"K={self.n_physical} physical qubits\n"
            f"Constraints: {len(self.plaquettes)}\n"
            f"Violated: {self.count_violated_constraints()}"
        )
        ax.text(
            0.02, 0.98,
            stats,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LHZArchitecture(N={self.n_logical} logical, "
            f"K={self.n_physical} physical, "
            f"{len(self.plaquettes)} constraints)"
        )

    def summary(self) -> str:
        """Return a human-readable summary of the architecture."""
        lines = [
            "=" * 50,
            "  LHZ Architecture Summary",
            "=" * 50,
            f"  Logical qubits (N)  : {self.n_logical}",
            f"  Physical qubits (K) : {self.n_physical}  [= N(N-1)/2]",
            f"  Plaquette constraints: {len(self.plaquettes)}  [= K - N + 1]",
            f"  Constraint strength : C/J = {self.constraint_strength}",
            "-" * 50,
            "  Physical Qubit Layout:",
        ]
        for q in self.physical_qubits:
            lines.append(
                f"    [{q.index:2d}] bond ({q.logical_i},{q.logical_j})  "
                f"J={q.coupling:+.3f}  state={q.state}"
            )
        lines.append(
            f"\n  Violated constraints: {self.count_violated_constraints()} / {len(self.plaquettes)}"
        )
        lines.append(f"  Total energy: {self.total_energy():.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)