"""
examples/maxcut_demo.py
=======================
Demonstrates the LHZ optimizer solving a Max-Cut problem.

Run with:
    python examples/maxcut_demo.py
"""

import networkx as nx
from lhz.optimizer import LHZOptimizer

def main():
    print("=" * 55)
    print("  LHZ Quantum Toolkit — Max-Cut Demo")
    print("=" * 55)

    # Build a random 3-regular graph on 6 nodes
    G = nx.random_regular_graph(3, 6, seed=7)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Max possible cut: {G.number_of_edges()} edges\n")

    # Build optimizer and solve
    optimizer = LHZOptimizer(G, constraint_strength=2.0, seed=42)
    print(optimizer.arch)

    print("\nRunning simulated annealing...")
    result = optimizer.solve(n_sweeps=2000, T_start=5.0, T_end=0.01)

    print(f"\nResults:")
    print(f"  Cut value    : {result.cut_value} edges")
    print(f"  Best energy  : {result.best_energy:.4f}")
    print(f"  Logical spins: {result.best_logical}")
    print(f"  Partition A  : {sorted(result.cut_partition[0])}")
    print(f"  Partition B  : {sorted(result.cut_partition[1])}")
    print(f"  Accept rate  : {result.acceptance_rate:.1%}")

    print("\nSaving solution plot → maxcut_solution.png")
    fig = result.visualize_solution(graph=G)
    fig.savefig("maxcut_solution.png", dpi=150, bbox_inches="tight")
    print("Done!")

if __name__ == "__main__":
    main()