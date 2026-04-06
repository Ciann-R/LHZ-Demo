"""
examples/error_tolerance_demo.py
=================================
Replicates Figure 3 from Lechner, Hauke, Zoller (2015).

Shows that despite the quadratic growth of physical qubits,
the total error probability scales only linearly with N.

Run with:
    python examples/error_tolerance_demo.py
"""

from lhz.error_analysis import ErrorAnalyzer

def main():
    print("=" * 55)
    print("  LHZ Error Tolerance Analysis (Fig. 3 replication)")
    print("=" * 55)
    print("\nAnalyzing fault tolerance for N = 2 to 8 logical qubits...")
    print("(This may take ~30 seconds for n_trials=500)\n")

    analyzer = ErrorAnalyzer(
        n_range=range(2, 9),
        n_trials=300,
        decoherence_rate=0.01,
        seed=42,
    )
    result = analyzer.run()

    print("\nSummary table:")
    print(f"{'N':>4} {'K':>6} {'P_d':>10} {'P_m':>10} {'P_d*P_m':>12} {'Readouts':>12}")
    print("-" * 58)
    for i, N in enumerate(result.n_values):
        print(
            f"{N:>4} {result.n_physical[i]:>6} "
            f"{result.p_spin_flip[i]:>10.4f} "
            f"{result.p_info_loss[i]:>10.4f} "
            f"{result.total_error[i]:>12.4f} "
            f"{result.n_readouts[i]:>12}"
        )

    print("\nSaving plot → error_tolerance.png")
    fig = result.plot()
    fig.savefig("error_tolerance.png", dpi=150, bbox_inches="tight")
    print("Done!")

if __name__ == "__main__":
    main()