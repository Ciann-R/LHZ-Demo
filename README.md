# LHZ Quantum Demo 

A Python implementation of the **Lechner-Hauke-Zoller (LHZ) parity architecture** for quantum annealing, based on:

> *"A quantum annealing architecture with all-to-all connectivity from local interactions"*
> W. Lechner, P. Hauke, P. Zoller — Science Advances, 2015. [DOI: 10.1126/sciadv.1500838](https://doi.org/10.1126/sciadv.1500838)

---

## What is this?

The LHZ architecture solves a core problem in quantum annealing: physical qubits only interact *locally*, but optimization problems often require *all-to-all* connectivity. This toolkit simulates and visualizes how the LHZ encoding bridges that gap.


| Module | What it does |
|---|---|
| `lhz.architecture` | Encodes N logical qubits → K physical qubits, builds constraint lattice |
| `lhz.optimizer` | Solves real problems (Max-Cut, TSP) using LHZ-encoded simulated annealing |
| `lhz.error_analysis` | Replicates Fig. 3 from the paper — fault tolerance vs. system size |

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/lhz-quantum-toolkit.git
cd lhz-quantum-toolkit
pip install -e ".[dev]"

# Run a demo
python examples/maxcut_demo.py
```

---

## Installation

```bash
pip install -e .          # Basic install
pip install -e ".[dev]"   # With dev tools (pytest, black, etc.)
```

**Requirements:** Python 3.9+, NumPy, Matplotlib, NetworkX

---

## Usage Examples

### 1. Build an LHZ Architecture

```python
from lhz.architecture import LHZArchitecture

# Create architecture for 4 logical qubits
arch = LHZArchitecture(n_logical=4)
print(f"Logical qubits: {arch.n_logical}")
print(f"Physical qubits: {arch.n_physical}")   # N(N-1)/2 = 6
print(f"Constraints: {arch.n_constraints}")    # K - N + 1 = 3

arch.visualize()  # Plot the constraint lattice
```

### 2. Solve a Max-Cut Problem

```python
import networkx as nx
from lhz.optimizer import LHZOptimizer

G = nx.random_regular_graph(3, 6, seed=42)
optimizer = LHZOptimizer(G)
result = optimizer.solve(n_sweeps=1000)
print(f"Best cut value: {result.cut_value}")
result.visualize_solution()
```

### 3. Analyze Error Tolerance

```python
from lhz.error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer(n_range=range(2, 10), n_trials=500)
analyzer.run()
analyzer.plot()  # Reproduces Fig. 3 from Lechner et al. 2015
```

---

## Repository Structure

```
lhz-quantum-toolkit/
├── lhz/
│   ├── __init__.py
│   ├── architecture.py     # Core LHZ encoding & constraint lattice
│   ├── optimizer.py        # Problem solver (Max-Cut, TSP)
│   └── error_analysis.py   # Fault tolerance analysis
├── examples/
│   ├── maxcut_demo.py
│   ├── tsp_demo.py
│   └── error_tolerance_demo.py
├── tests/
│   ├── test_architecture.py
│   ├── test_optimizer.py
│   └── test_error_analysis.py
├── notebooks/
│   └── lhz_walkthrough.ipynb
├── docs/
│   └── theory.md
├── .github/workflows/
│   └── tests.yml
├── pyproject.toml
└── README.md
```

---

## Background: 

In standard quantum annealing, you need to control *J_ij* , the interaction between every pair of logical spins. That's O(N²) couplers, which is physically hard to wire up.

LHZ flips the model:
- Each **physical qubit** represents a *bond* between two logical qubits (parallel=1, antiparallel=0)
- The **optimization parameters** J_ij become *local fields* on physical qubits (easy to control!)
- **Parity constraints** on 4-qubit plaquettes enforce logical consistency

N logical qubits → N(N-1)/2 physical qubits. But you gain full programmability and intrinsic fault tolerance.



```bibtex
@article{lechner2015quantum,
  title={A quantum annealing architecture with all-to-all connectivity from local interactions},
  author={Lechner, Wolfgang and Hauke, Philipp and Zoller, Peter},
  journal={Science Advances},
  volume={1},
  number={9},
  pages={e1500838},
  year={2015},
  publisher={American Association for the Advancement of Science}
}
```