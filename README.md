# Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones
<p align="center">
  <img src="logo_assets/Logo.jpeg" alt="Logo" width="100"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![Framework](https://img.shields.io/badge/Type-Agent--Based%20Simulation-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![DEAP](https://img.shields.io/badge/DEAP-GitHub-0A66C2.svg)](https://github.com/deap/deap)
[![PySwarms](https://img.shields.io/badge/PySwarms-PyPI-1F6FEB.svg)](https://pypi.org/project/pyswarms/)

## Overview

This repository accompanies the GECCO 2026 paper:

> **Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones**  
> *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2026)*

The repository implements the **Production Tax Zone Model (PTZM)**, an anticipatory computational framework for studying the emergence of tax-minimizing strategies in production environments with heterogeneous tax treatment.

PTZM combines:

- agent-based modeling of heterogeneous firms,
- discrete simulation of production and trade,
- accounting-aware state transitions, and
- heuristic optimization over feasible transaction structures.

Rather than evaluating a single predefined abuse pattern, the framework treats tax systems as strategic environments in which transaction sequences can emerge endogenously under legal, production, and accounting constraints.

---

## Motivation

Conventional tax analysis is often static, case-specific, and reactive. By contrast, PTZM is designed as a computational laboratory for **ex-ante policy stress testing**.

This makes it possible to:

- discover tax-minimizing behaviors without fully pre-specifying them in advance;
- model the interaction between production structure, accounting rules, and tax incentives;
- analyze how changes in legal design alter the space of feasible and attractive strategies; and
- explore the anticipatory use of evolutionary computation in regulatory analysis.

---

## Key Contributions

This repository implements a research framework with the following core contributions:

- **Production-constrained tax environments** formalized through directed acyclic graphs (DAGs);
- **accounting-aware simulation**, including firm-level bookkeeping and financial-state updates;
- **endogenous generation of feasible transaction sequences** under legal and technological constraints;
- **heuristic search over tax-minimizing strategy spaces**, including Genetic Algorithm (GA), Binary PSO, and Random Search;
- **policy-grounded experimentation** inspired by the Colombian Free Trade Zone (FTZ) regime, including pre- and post-2022 institutional settings.

---

## Repository Structure

```bash
├── configs/
│   ├── chart_of_accounts/
│   ├── experiment_configs/
│   └── fine_tuning_configs/
├── docs/
│   └── experiments.md
├── paper_results/
│   ├── docs/
│   ├── experiment/
│   ├── fine_tuning/
│   └── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   ├── experiment/
│   ├── fine_tuning/
│   ├── simulation/
│   └── utilities/
└── README.md
```

### Main directories

- `src/`: core implementation of the simulation, search procedures, and experiment pipelines.
- `configs/`: chart of accounts, benchmark configuration files, graph suites, and fine-tuning inputs.
- `paper_results/`: documentary and analytical artifacts associated with the paper.
- `docs`: detailed operational documentation for benchmark execution and fine-tuning workflows.

---

## Model at a Glance

### Economic Environment

The framework represents production and taxation through a structured environment that includes:

- a **production graph (DAG)** defining technological input-output constraints;
- **heterogeneous firms** operating under different tax regimes;
- **assets, inventories, and ownership tracking**;
- **accounting states** that evolve with production and trade;
- **exogenous price tensors** governing inter-entity transactions; and
- **tax rules** attached to economic events.

### Simulation Layer

The simulation executes sequences of economic transactions and updates:

- firm revenues,
- firm costs,
- tax liabilities,
- inventories,
- asset costs, and
- final after-tax profits.

This allows candidate strategies to be evaluated as internally consistent economic processes rather than abstract combinatorial assignments.

### Optimization Layer

Strategies are encoded as binary vectors and mapped to feasible transaction sequences. The objective is to maximize the after-tax profit of a coalition of firms operating across tax jurisdictions.

Supported search methods currently include:

| Method | Implementation |
|---|---|
| Genetic Algorithm | DEAP-based |
| Particle Swarm Optimization | Custom Implementation based on Pyswarms |
| Binary Random Search | Baseline |

Reference libraries:

[![DEAP](https://img.shields.io/badge/DEAP-GitHub-0A66C2.svg)](https://github.com/deap/deap)
[![PySwarms](https://img.shields.io/badge/PySwarms-PyPI-1F6FEB.svg)](https://pypi.org/project/pyswarms/)

---

## Colombian FTZ-Inspired Policy Setting    

The implementation is institutionally inspired by the Colombian Free Trade Zone (FTZ) regime.

In particular:

- the **accounting** structure is grounded in Colombian-style accounting categories;
- the **tax differentials** are motivated by the Colombian FTZ framework; and
- the experimental setting reflects the contrast between two regulatory regimes:
- **Pre-2022**: strategy space shaped by differential income tax rates between FTZ and standard firms;
- **Post-2022**: export-contingent benefit structure introduced through regulatory reform.

This setting allows direct analysis of how changes in institutional design reshape the landscape of emergent strategies.

---

## Quick Start

### Installation

We recommend using **Conda** for environment isolation.

```bash
# 1. Clone the repository
git clone https://github.com/Tax-Avoidance-Detection-MIT-Global-Seed/evolutionary-tax-strategies-sez.git
cd repo

# 2. Create and activate environment
conda create -n ptzm python=3.9 -y
conda activate ptzm

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Minimal Reproduction

A minimal validation run of the benchmark pipeline can be launched with:

```bash
python src/experiment/run_experiment.py --algos generic --runs 1 --gens 5 --max-workers 1
```
This command is intended as a first-pass execution test. It verifies that the benchmark orchestration pipeline, environment generation, and algorithm wrapper are correctly wired.

---

## Main Entry Points

### Benchmark experiments
```bash
src/experiment/run_experiment.py
```

Primary inputs:
```bash
configs/experiment_configs/algorithms_config/exp_config.json
configs/experiment_configs/input_graphs/test_suite_graphs.json
configs/chart_of_accounts/chart_of_accounts.yaml
```

### Fine-tuning workflow
```bash
src/fine_tuning/scripts/run_all.py
```

Primary inputs typically include:
```bash
configs/fine_tuning_configs/parameter_samples/plan.json
configs/fine_tuning_configs/parameter_samples/samples/
configs/chart_of_accounts/chart_of_accounts.yaml
```

---

## Project Documentation

Detailed documentation is organized separately from this main README.

### Experimental pipelines and execution details

For the full benchmark workflow, configuration logic, output structure, reproducibility notes, and the fine-tuning pipeline, see:


[docs/experiments.md](docs/experiments.md)


### Paper artifacts and result navigation

For notebooks, PDFs, benchmark-environment documentation, ladder-strategy materials, benchmark-analysis notebooks, and analytical support artifacts, see:




This separation is intentional: the main README provides a high-level entry point, while the linked documents provide exhaustive operational and documentary detail.

---

## Paper Artifacts

The repository includes the main documentary artifacts used to support the GECCO 2026 paper, including:

- benchmark DAG documentation,
- environment-analysis notebooks,
- ladder-strategy reconstruction materials,
- benchmark-result analysis notebooks, and
- fine-tuning inspection artifacts and parameter-selection notebooks.

A guided overview of these materials is provided in:


[paper_results/README.md](paper_results/README.md)

### Ladder Strategy Materials

The main notebook and PDF corresponding to the paper’s *Ladder Strategy* analysis are available here:

- [`ladder_strategy.ipynb`](paper_results/docs/ladder_strategy/ladder_strategy.ipynb)
- [`ladder_strategy.pdf`](paper_results/docs/ladder_strategy/ladder_strategy.pdf)

These files provide both the executable and reading-oriented versions of the paper’s central strategic result.

---

### Benchmark Test-Suite Documentation

The notebook and PDF documenting the benchmark DAG suite used in the paper are available here:

- [`Graphs_prop.ipynb`](paper_results/docs/environment_description/Graphs_prop.ipynb)
- [`DAG_Structures.pdf`](paper_results/docs/environment_description/DAG_Structures.pdf)

These materials describe the structure, design rationale, and interpretation of the benchmark environments used throughout the experimental analysis.

---

## Reproducibility Notes

- Benchmark runs are best launched from a terminal using the Python entry-point scripts.
- On Windows systems, multiprocessing is generally more reliable when invoked from a standard script execution context rather than from notebooks or interactive shells.
- A short validation run is recommended before launching full benchmark suites.
- Full benchmark and fine-tuning documentation is available in `docs/experiments.md`.

---

## Extending the Framework

The repository separates model logic, search procedures, and execution pipelines.

In broad terms, adding a new search method requires:

1. implementing the method in the relevant source module;
2. exposing an experiment- or fine-tuning-level wrapper when needed; and
3. registering the method in the relevant configuration files.

This modular structure is intended to support future methodological extensions while preserving a stable benchmarking workflow.

---

## Citation

Pending to Change as DOI activation proceeds

```bibtex
@inproceedings{ptzm2026evolutionary,
  title     = {Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones},
  author    = {Andres Leguizamon and Carlos David Sanchez and Sof{\\'i}a Ocampo and Una-May O'Reilly and Erik Hemberg},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2026)},
  year      = {2026},
  publisher = {ACM}
}
```
