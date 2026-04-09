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

```bash
├── configs/
│   ├── chart_of_accounts/
│   ├── experiment_configs/
│   └── fine_tuning_configs/
├── paper_results/
│   ├── docs/
│   ├── experiment/
│   └── fine_tuning/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   ├── experiment/
│   ├── fine_tuning/
│   ├── simulation/
│   └── utilities/
```


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

## Colombian FTZ-Inspired Policy Setting

The implementation is inspired by the Colombian Free Trade Zone (FTZ) regime. In particular, both the accounting structure and the tax parameters are grounded in the Colombian context: the accounting categories are based on Colombian-style accounts, and the tax-rate differentials are motivated by the Colombian FTZ framework.

- **Pre-2022:** Strategy space shaped by differential income tax rates between FTZ and standard firms
- **Post-2022:** Export-contingent benefit structure introduced by regulatory reform

This two-regime setup enables direct analysis of how changes in the legal and tax framework reshape the landscape of emergent strategies.
---

## Quick Start

### Installation
We recommend using **Conda** for environment isolation to ensure dependency versions (such as DEAP or network libraries) do not conflict with your base system.

```bash
# 1. Clone the repository
git clone [https://github.com/******.git](https://github.com/********.git)
cd repo

# 2. Create and activate environment
conda create -n ptzm python=3.9 -y
conda activate ptzm

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

The benchmark experiment pipeline is defined by two primary inputs:

- `configs/experiment_configs/algorithms_config/exp_config.json`
- `configs/experiment_configs/input_graphs/test_suite_graphs.json`

The corresponding execution entry point is:

- `src/experiment/run_experiment.py`

Within this pipeline, the repository:

1. parses the algorithm and environment specification;
2. loads the production-graph suite;
3. generates standard and perturbed environments;
4. attaches the base agent definitions to each environment;
5. resolves the canonical chart of accounts at `configs/chart_of_accounts/chart_of_accounts.yaml`;
6. executes the selected optimization procedures across all generated environments; and
7. materializes logs, metadata, intermediate environment tables, and final result files in a timestamped output directory.

#### Configuration Schema

The operational configuration is stored in JSON. The following YAML representation is included solely as a schema-oriented template, because it exposes the logical structure more transparently than raw JSON. The file that governs execution remains:

- `configs/experiment_configs/algorithms_config/exp_config.json`

The complete YAML-style template is provided in the expandable block below.

<details>
<summary>View YAML configuration template</summary>

```yaml
common:
  fix_last_gene: true
  seed: 42
  verbosity: 2
  log_every: 1
  evals_cap: 10000
  time_limit_sec: null
  notes: "Global parameters shared by all algorithms"

env_construction:
  base: 100
  m1: 0.01
  m2: 0.1
  m3: 0.25
  n_agents: 3
  perturb_std: 0.1
  ignored_agents: [0]
  alfa: 4
  rho: 1.15
  min_budget: 1500
  max_budget: 10000
  notes: "Parameters used for price generation and budget estimation"

flat:
  parents_rate: 0.3
  mutation_rate: 0.05

generic:
  cxpb: 0.8266
  mutpb: 0.4193
  mutation_rate: 0.01054
  elitism: 1

joint:
  mode: graph
  parents_rate: 0.3
  sel_mutation: 0.25
  tail_mutation: 0.02
  per_good_cap: null
  max_index_probe: 8

pso:
  c1: 1.4243
  c2: 2.1965
  w: 0.9689

random:
  evals_cap: 5000
  time_limit_sec: null
  fix_last_gene: true

macro_micro:
  mode: graph
  per_good_cap: null
  max_index_probe: 8
  parents: 20
  elite_fraction: 0.01
  tourn_size: 5
  parent_selection: tournament
  mating_selection: pairwise_tournament
  lambda_in: 0.30
  lambda_out: 0.30
  p_macro: 1.0
  p_micro: 1.0
  sel_mutation: 0.065
  tail_mutation: 0.02
  p_min: 0.3
  tau_percent: 0.05

macro:
  mode: graph
  per_good_cap: null
  max_index_probe: 8
  parents: 20
  elite_fraction: 0.01
  tourn_size: 5
  parent_selection: tournament
  mating_selection: pairwise_tournament
  lambda_in: 0.4
  lambda_out: 0.3
  p_macro: 1.0
  p_micro: 0.0
  sel_mutation: 0.05
  tail_mutation: 0.05
  p_min: 0.3
  tau_percent: 0.05

micro:
  mode: graph
  per_good_cap: null
  max_index_probe: 8
  parents: 20
  elite_fraction: 0.01
  tourn_size: 5
  parent_selection: tournament
  mating_selection: pairwise_tournament
  lambda_in: 0.4
  lambda_out: 0.3
  p_macro: 0.0
  p_micro: 1.0
  sel_mutation: 0.02
  tail_mutation: 0.02
  p_min: 0.3
  tau_percent: 0.05

recomb:
  mode: graph
  per_good_cap: null
  max_index_probe: 8
  parents: 20
  elite_fraction: 0.01
  tourn_size: 5
  parent_selection: tournament
  mating_selection: pairwise_tournament
  p_recomb: 0.5
  sel_mutation: 0.15
  tail_mutation: 0.035
  p_min: 0.3
  tau_percent: 0.05

no_crossover:
  mode: graph
  per_good_cap: null
  max_index_probe: 8
  parents: 20
  elite_fraction: 0.01
  tourn_size: 5
  parent_selection: tournament
  mating_selection: pairwise_tournament
  lambda_in: 0.25
  lambda_out: 0.5
  p_macro: 0.0
  p_micro: 0.0
  sel_mutation: 0.05
  tail_mutation: 0.05
  p_min: 0.3
  tau_percent: 0.05

mixed_generic:
  cxpb: 0.731
  mutpb: 0.216
  mutation_rate: 0.0094
  selector_mutation_rate: 0.095
  elitism: 1

chart_of_accounts:
  yaml_name: ../../chart_of_accounts/chart_of_accounts.yaml

BASE_AGENTS:
  MKT:
    type: MKT
    inventory_strategy: FIFO
    firm_related_goods: []
    income_statement_type: standard
    accounts_yaml_path: null
    price_mapping: 0
  NCT:
    type: NCT
    inventory_strategy: FIFO
    firm_related_goods: []
    income_statement_type: standard
    accounts_yaml_path: null
    price_mapping: 1
  ZF:
    type: ZF
    inventory_strategy: FIFO
    firm_related_goods: []
    income_statement_type: standard
    accounts_yaml_path: null
    price_mapping: 2
```

</details>

#### Interpretation Of Configuration Blocks

- `common`: hyperparameters shared by all algorithms unless explicitly overridden within an algorithm-specific block.
- `env_construction`: parameters governing price generation, perturbation, and budget estimation for each graph-derived environment.
- algorithm blocks such as `generic`, `flat`, `joint`, `pso`, `random`, `macro_micro`, `macro`, `micro`, `recomb`, `no_crossover`, and `mixed_generic`: procedure-specific hyperparameter definitions.
- `chart_of_accounts`: relative reference to the accounting specification used by the agents. Path resolution is normalized internally to the canonical file under `configs/chart_of_accounts/chart_of_accounts.yaml`.
- `BASE_AGENTS`: base agent templates replicated into each generated environment.

#### Configuration Notes

- The graph suite is maintained separately from the algorithm configuration.
- Each graph generally induces multiple environments, most notably standard and perturbed variants.
- The computational burden of a full benchmark scales approximately as:

```text
number_of_environments x number_of_algorithms x runs
```

- In `run_test_env.py`, population size is derived from:

```text
budget / generations
```

Consequently, reducing the number of generations does not necessarily produce a cheap run if the environment budget remains large.

- The chart of accounts should reference the canonical file:

```text
configs/chart_of_accounts/chart_of_accounts.yaml
```

### Running an Experiment

#### Validation Run

The following command provides a minimal validation of the non-fine-tuning benchmark pipeline:

```bash
python src/experiment/run_experiment.py --algos generic --runs 1 --gens 5 --max-workers 1
```

Interpretation of the command:

- `python src/experiment/run_experiment.py`: executes the benchmark orchestration script.
- `--algos generic`: restricts execution to the generic genetic algorithm wrapper.
- `--runs 1`: performs one repetition per generated environment.
- `--gens 5`: sets the evolutionary horizon to five generations.
- `--max-workers 1`: disables parallel fan-out beyond a single worker and is therefore suitable for first-pass validation.

#### Full Benchmark Invocation

```bash
python src/experiment/run_experiment.py --algos flat,generic,joint,random,pso,macro_micro,macro,micro,recomb,no_crossover,mixed_generic --runs 6 --gens 100 --seed 42 --max-workers 4 --graphs configs/experiment_configs/input_graphs/test_suite_graphs.json --config configs/experiment_configs/algorithms_config/exp_config.json --chart configs/chart_of_accounts/chart_of_accounts.yaml --output ./exp_output --tag baseline
```

Interpretation of the principal arguments:

- `--algos`: comma-separated subset of algorithms to execute; each identifier must be present in the configuration file.
- `--runs`: number of repetitions per algorithm and per environment.
- `--gens`: number of generations allocated to each run.
- `--seed`: base seed from which environment- and run-specific seeds are derived.
- `--max-workers`: number of worker processes used by the multiprocessing executor.
- `--graphs`: graph JSON file defining the production DAG suite.
- `--config`: experiment configuration JSON containing algorithm blocks, environment settings, and base agent definitions.
- `--chart`: path to the chart-of-accounts YAML file.
- `--output`: root directory under which timestamped experiment folders are created.
- `--tag`: optional identifier appended to the output-folder name.

#### Output Files

Each benchmark invocation creates a timestamped directory under the designated output root. The principal artifacts are:

- `run_log.txt`
- `metadata.json`
- `config_used.json`
- `environment_database.csv`
- `results.csv`

#### Reproducibility Notes

- Benchmark runs should be launched from a terminal using the `.py` entry point.
- On Windows, multiprocessing is substantially more reliable when the main process is started from a real script rather than from an interactive shell or notebook cell.
- A short validation run is advisable before launching the full benchmark suite.
- The shortest validation path is obtained by restricting execution to `--algos generic`.
- Runs that include `pso` should be launched from a writable working directory, because some dependencies may emit auxiliary log files during execution.


### Fine-Tuning Pipeline

The repository also includes a distinct fine-tuning workflow for hyperparameter search on a fixed environment. This pipeline is methodologically different from the benchmark experiment pipeline described above.

The main entry point is:

- `src/fine_tuning/scripts/run_all.py`

Its principal inputs are:

- a plan file, typically `configs/fine_tuning_configs/parameter_samples/plan.json`
- a candidates directory, typically `configs/fine_tuning_configs/parameter_samples/samples`
- an output directory where per-algorithm results will be written

#### Fine-Tuning Plan Structure

```yaml
seeds: [101, 202, 303, 404]
env_json: ../fine_tuning_env_config/prices/FT2_Prices_V1_env.json
accounts_yaml: ../../chart_of_accounts/chart_of_accounts.yaml
algorithms: [generic, mixed_generic, pso, joint, baseline, macro, micro, macro_micro, recombination]
generations: 100
popsize: 26
evals_cap: 10000
n_samples_per_algo: 100
bounds:
  generic:
    cxpb: [0.01, 0.99]
    mutpb: [0.01, 0.50]
    mutation_rate: [0.01, 0.20]
```

Interpretation of the principal fields:

- `seeds`: repeated evaluation seeds used to assess candidate robustness.
- `env_json`: environment specification containing the production graph and price tensor.
- `accounts_yaml`: chart-of-accounts file attached to the environment agents.
- `algorithms`: algorithms included in the fine-tuning pass.
- `generations`, `popsize`, `evals_cap`: shared base hyperparameters for the fine-tuning run.
- `n_samples_per_algo`: nominal number of sampled candidates per algorithm.
- `bounds`: continuous ranges or discrete grids defining each algorithm's admissible hyperparameter space.

Each algorithm additionally requires a candidate file in the candidates directory, for example:

- `candidates_generic.json`
- `candidates_joint.json`
- `candidates_pso.json`

These files contain candidate identifiers and the corresponding hyperparameter dictionaries to be evaluated.

#### Fine-Tuning Command

```bash
python src/fine_tuning/scripts/run_all.py --plan configs/fine_tuning_configs/parameter_samples/plan.json --candidates-dir configs/fine_tuning_configs/parameter_samples/samples --out-base ./ft_output --algos generic,joint,pso --n-jobs 4 --emit-csv
```

Interpretation of the principal arguments:

- `--plan`: fine-tuning plan JSON with seeds, environment path, accounting path, and shared computational budgets.
- `--candidates-dir`: directory containing `candidates_<algo>.json` files.
- `--out-base`: base output directory for per-algorithm result folders.
- `--algos`: subset of fine-tuning algorithms to execute.
- `--n-jobs`: number of local worker processes.
- `--emit-csv`: requests per-algorithm CSV summaries in addition to the per-run JSON files.

#### Fine-Tuning Outputs

The fine-tuning pipeline writes:

- one JSON result per `(algorithm, candidate, seed)`
- one directory per algorithm under the selected output base
- optional per-algorithm CSV summaries when `--emit-csv` is enabled

The separation between `experiment` and `fine_tuning` is intentional. The former benchmarks algorithms across an environment suite; the latter evaluates candidate hyperparameter sets on a fixed environment under a controlled seed schedule.

---

## Paper Results

The repository contains the principal documentary artifacts used to explain the benchmark environments, the price-construction mechanism, and the emergence of the `Ladder Strategy` reported in the paper.

The recommended order of consultation is:

1. the environment-description materials;
2. the source code responsible for price generation; and
3. the `Ladder Strategy` notebook together with its PDF counterpart.

### Environment Explanation

The complete explanation of the benchmark DAG structures used in the test suite is documented here:

```text
paper_results/docs/environment_description/DAG_Structures.pdf
```

This document provides the primary reference for:

- the production-graph families used in the experiments;
- the structural differences between graph classes; and
- the logic underlying the benchmark environment suite.

The same directory also contains a notebook with environment-level analysis:

```text
paper_results/docs/environment_description/Graphs_prop.ipynb
```

Together, the PDF and notebook constitute the formal explanation of the test suite. The environment suite is therefore not a collection of ad hoc graphs; it is an explicitly analyzed and documented benchmark family.

### Price Construction

Synthetic prices for the benchmark environments are generated in the experiment pipeline. The principal implementation resides in:

```text
src/experiment/exp_configuration.py
```

The key function is `generate_prices(...)`, which constructs the standard and perturbed price tensors attached to each experiment environment.

That function relies on:

```text
src/simulation/economy/order_book/utils/price_markup_generator.py
```

From a methodological standpoint:

- environment assembly logic: `src/experiment/exp_configuration.py`
- price tensor generation logic: `src/simulation/economy/order_book/utils/price_markup_generator.py`

### Ladder Strategy

The central interpretive artifact for the paper's main strategic result is:

```text
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
```

This notebook is one of the repository's principal research artifacts. It provides the most direct reconstruction of how the `Ladder Strategy` emerges from the model and how that strategy is interpreted in the paper.

The PDF companion is here:

```text
paper_results/docs/ladder_strategy/ladder_strategy.pdf
```

Taken together, these two files provide the most compact route to the paper's central strategic result:

- executable analysis: `paper_results/docs/ladder_strategy/ladder_strategy.ipynb`
- stable reading and citation artifact: `paper_results/docs/ladder_strategy/ladder_strategy.pdf`

These materials should be read together with the environment documentation, because the `Ladder Strategy` is inseparable from the graph structures and price system from which it emerges.

### Parameter Selection And Experiment Analysis

The repository also preserves the analytical notebooks used to inspect the benchmark results and to select heuristic hyperparameters after the fine-tuning stage.

The principal experiment-level analysis notebook is:

```text
paper_results/experiment/experiment_analysis.ipynb
```

This notebook documents the post-processing and comparative analysis of the benchmark experiment outputs.

The fine-tuning results and the corresponding heuristic-parameter selection notebooks are organized by algorithm under:

```text
paper_results/fine_tuning/individual_algorithm_results/
```

The algorithm-specific parameter-selection notebooks currently included in the repository are listed in the expandable block below.

<details>
<summary>View algorithm-specific parameter-selection notebooks</summary>

```text
paper_results/fine_tuning/individual_algorithm_results/baseline/baseline_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/generic/generic_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/joint/joint_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/macro/macro_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/macro_micro/macro_micro_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/micro/micro_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/mixed_generic/mixed_generic_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/pso/pso_best_params.ipynb
paper_results/fine_tuning/individual_algorithm_results/recomb/recomb_best_params.ipynb
```

</details>

These notebooks constitute the documentary record of how the fine-tuning outputs were inspected and how the final heuristic configurations were selected for the reported experiments.

### Recommended Navigation Paths

For the benchmark suite:

```text
paper_results/docs/environment_description/DAG_Structures.pdf
paper_results/docs/environment_description/Graphs_prop.ipynb
```

For price construction in code:

```text
src/experiment/exp_configuration.py
src/simulation/economy/order_book/utils/price_markup_generator.py
```

For the main strategic result of the paper:

```text
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
paper_results/docs/ladder_strategy/ladder_strategy.pdf
```

For benchmark-result analysis and heuristic-parameter selection:

```text
paper_results/experiment/experiment_analysis.ipynb
paper_results/fine_tuning/individual_algorithm_results/
```

---

## Extending the Framework

The repository separates algorithmic logic from execution orchestration. In practical terms, extension requires both an implementation layer and a registration layer.

To add a new search procedure:

1. implement the search logic in the appropriate module under `src/search_heuristics/` or the corresponding wrapper layer;
2. expose an experiment-level wrapper under `src/experiment/wrappers/` and, if relevant, a fine-tuning wrapper under `src/fine_tuning/wrappers/`; and
3. register the procedure in the relevant configuration files so that it becomes callable from the experiment or fine-tuning entry points.

---

## Citation

```bibtex
@article{yourpaper,
  title   = {Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones},
  author  = {...},
  year    = {2026}
}
```
