"""
examples/architecture_demo.py
==============================
Demonstrates the LHZ physical qubit lattice for N=4 and N=5 logical qubits.

Run with:
    python examples/architecture_demo.py
"""

import numpy as np
from lhz.architecture import LHZArchitecture

def main():
    print("=" * 55)
    print("  LHZ Architecture Demo")
    print("=" * 55)

    for N in [4, 5]:
        print(f"\n--- N = {N} logical qubits ---")

        # Use the same coupling matrix as the paper (random J_ij ~ Uniform[-J, J])
        rng = np.random.default_rng(42)
        J = rng.uniform(-1.0, 1.0, (N, N))
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)

        arch = LHZArchitecture(n_logical=N, coupling_matrix=J, constraint_strength=2.0)
        print(arch.summary())

        fig = arch.visualize(title=f"LHZ Architecture — N={N} logical qubits")
        fig.savefig(f"architecture_N{N}.png", dpi=150, bbox_inches="tight")
        print(f"Saved → architecture_N{N}.png")

    print("\nDone!")

if __name__ == "__main__":
    main()
