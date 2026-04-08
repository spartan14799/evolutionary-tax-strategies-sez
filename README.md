# Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![Framework](https://img.shields.io/badge/Type-Agent--Based%20Simulation-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

This repository accompanies the paper:

> **Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones**
> *[Conference / Journal — TBD]*

We introduce the **Production Tax Zone Model (PTZM)**, an anticipatory computational framework that studies the emergence of tax-minimizing strategies in complex regulatory environments. The model combines agent-based modeling of heterogeneous firms, discrete simulation of production and trade, and heuristic optimization — enabling a computational laboratory for tax policy evaluation rather than a single predefined scenario.

---

## Motivation

Standard tax analysis is static and rule-based. PTZM instead treats tax systems as environments where strategies can emerge endogenously. This allows us to:

- Discover tax-minimizing behaviors without pre-specifying them
- Model the interplay between legal structure, accounting rules, and production constraints
- Run controlled experiments on how regulatory changes affect firm behavior

---

## Key Contributions

- Formalization of production-constrained tax environments using directed acyclic graphs (DAGs)
- Firm-level double-entry bookkeeping integrated into agent-based simulation
- Endogenous generation of feasible transaction sequences under legal constraints
- Comparative evaluation of evolutionary algorithms (GA, PSO, Random Search) for strategy optimization
- Empirical application to Colombian Free Trade Zone (FTZ) regulation, covering pre- and post-2022 reforms

---

## Architecture

├── configs/
│   ├── chart_of_accounts/
│   ├── experiment_configs/
│   └── fine_tuning_configs/
├── paper_results/
│   ├── docs/
│   ├── experiment/
│   └── fine_tuning/
├── pytest.ini
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   ├── experiment/
│   ├── fine_tuning/
│   ├── simulation/
    └── utilities/


---

## Model Components

### Economic Environment

- **Production Graphs (DAGs):** Enforce technological feasibility across goods and inputs.
- **Heterogeneous Firms:** Agents differ in tax regimes, accounting rules, and cost structures.
- **Double-Entry Accounting:** Full transaction-level bookkeeping with firm-level financial reporting (income statement, ledgers).

### Trade & Production

- Order-based trading between firms with a flexible, buyer/seller-dependent price system
- Input consumption governed by DAG structure, with cost propagation through production chains
- Explicit tracking of ownership, costs, and tax implications at each transaction

### Optimization Layer

Strategies are encoded as binary vectors and mapped to feasible sequences of economic transactions. The fitness function is the after-tax profit of a coalition of firms. Supported search methods:

| Method | Implementation |
|---|---|
| Genetic Algorithm | DEAP-based |
| Particle Swarm Optimization | Custom PSO |
| Binary Random Search | Baseline |

---

## Policy Application: Colombian Free Trade Zones

The implementation focuses on Colombian tax law and the Free Trade Zone (FTZ) regime:

- **Pre-2022:** Strategy space shaped by differential income tax rates between FTZ and standard firms
- **Post-2022:** Export-contingent benefit structure introduced by regulatory reform

This two-regime setup enables direct measurement of how rule changes alter the emergent strategy landscape.

---

## Quick Start

### Requirements

- Python 3.9+
- `pip install -r requirements.txt`

### Installation

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```

### Configuration

Experiments are configured via files in `config/`. Each configuration specifies:

**Environment inputs**
- Production graph (DAG)
- Price matrix
- Agent definitions
- Chart of accounts

**Algorithm parameters**
- Population size
- Mutation and crossover rates
- Evaluation budget

### Running an Experiment

```bash
# Example — to be updated with actual entry point
python main.py --config config/your_experiment.yaml
```

---

## Results

The test suite includes benchmark graphs of varying complexity and synthetic price generation mechanisms. Main results cover:

- Algorithm performance comparison across GA, PSO, and Random Search
- Conditions under which tax-minimizing strategies emerge
- Structural drivers of strategy formation in the Colombian FTZ case study

See `data_analysis/` for figures and result files.

---

## Extending the Framework

To add a new search algorithm:

1. Implement it inside `algorithms/`, following the existing evaluation interface
2. Register it in your experiment configuration file

---

## Citation

```bibtex
@article{yourpaper,
  title   = {Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones},
  author  = {...},
  year    = {2026}
}
```