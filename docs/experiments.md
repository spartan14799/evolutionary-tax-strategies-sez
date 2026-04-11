# Experiments and Fine-Tuning Guide

This document provides the operational and reproducibility-oriented documentation for the experimental workflows used in the GECCO 2026 paper:

> **Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones**

Its purpose is to document, in a single place, how benchmark experiments and fine-tuning runs are configured, launched, interpreted, and reproduced.

This guide complements, rather than replaces:

- the repository-level overview in `README.md`, and
- the paper-artifact guide in `paper_results/README.md`.

In short:

- `README.md` explains the project,
- `paper_results/README.md` explains the research artifacts,
- `docs/experiments.md` explains how the experimental machinery works.

---

## Scope of This Document

This file covers three main components:

1. the **benchmark experiment pipeline**, used to evaluate search procedures across a suite of production environments;
2. the **fine-tuning pipeline**, used to explore and select algorithm hyperparameters on a fixed environment; and
3. the **reproducibility conventions** associated with execution, configuration, and output interpretation.

This document is intentionally exhaustive at the operational level.

---

## Repository Paths Relevant to Experiments

The main experiment-related paths are:

```text
configs/experiment_configs/algorithms_config/exp_config.json
configs/experiment_configs/input_graphs/test_suite_graphs.json
configs/chart_of_accounts/chart_of_accounts.yaml
src/experiment/run_experiment.py
src/fine_tuning/scripts/run_all.py
```

Additional experiment-facing artifacts are documented in:

```text
paper_results/README.md
```

---

## Experimental Workflows at a Glance

The repository supports two distinct workflows.

### 1. Benchmark experiment workflow

This workflow evaluates one or more algorithms across a suite of graph-derived environments.

Its goals are to:

- generate benchmark environments from graph specifications;
- attach accounting and pricing structures to those environments;
- run search procedures under a shared evaluation protocol; and
- collect comparable outputs across algorithms and environments.

### 2. Fine-tuning workflow

This workflow evaluates candidate hyperparameter configurations on a fixed environment.

Its goals are to:

- sample or load candidate configurations;
- evaluate them repeatedly across a controlled seed schedule; and
- support the selection of final parameter settings for benchmark use.

These two workflows are methodologically different and should not be conflated.

---

# Part I. Benchmark Experiment Pipeline

## Benchmark Entry Point

The main script for benchmark execution is:

```text
src/experiment/run_experiment.py
```

This script orchestrates the full non-fine-tuning benchmark workflow.

---

## Benchmark Inputs

The benchmark pipeline is defined by three primary inputs:

```text
configs/experiment_configs/algorithms_config/exp_config.json
configs/experiment_configs/input_graphs/test_suite_graphs.json
configs/chart_of_accounts/chart_of_accounts.yaml
```

### Role of each input

- `exp_config.json`: experiment-level configuration, including shared parameters, environment-construction settings, algorithm blocks, and base-agent templates.
- `test_suite_graphs.json`: graph suite defining the production DAGs from which benchmark environments are generated.
- `chart_of_accounts.yaml`: canonical accounting specification attached to the experiment agents.

---

## Benchmark Pipeline Logic

At a high level, the benchmark experiment pipeline performs the following steps:

1. parses the algorithm and environment specification;
2. loads the production-graph suite;
3. generates standard and perturbed environments from those graphs;
4. attaches the base agent definitions to each environment;
5. resolves the canonical chart of accounts;
6. executes the selected optimization procedures across all generated environments; and
7. materializes logs, metadata, intermediate environment tables, and final result files in a timestamped output directory.

This process yields a benchmark layer that is graph-driven, configuration-driven, and reproducible through explicit file inputs.

---

## Configuration Structure

The operational configuration for the benchmark pipeline is stored in:

```text
configs/experiment_configs/algorithms_config/exp_config.json
```

Because the logic of the file is easier to read in hierarchical form, the following YAML-style representation can be used as a schema-oriented template.

The executable file remains JSON; the YAML below is for documentation only.

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

---

## Interpretation of Configuration Blocks

### `common`
This block contains parameters shared by all algorithms unless explicitly overridden inside an algorithm-specific block.

Typical examples include:

- global seed,
- evaluation cap,
- verbosity settings,
- logging frequency,
- whether the last gene is fixed,
- optional time limits.

### `env_construction`
This block governs environment generation and budget estimation.

It includes parameters related to:

- base price construction,
- markups,
- perturbation intensity,
- number of agents,
- ignored price mappings,
- budget-scaling parameters,
- minimum and maximum evaluation budgets.

### Algorithm blocks
Blocks such as `generic`, `flat`, `joint`, `pso`, `random`, `macro_micro`, `macro`, `micro`, `recomb`, `no_crossover`, and `mixed_generic` contain procedure-specific hyperparameters.

These blocks define the admissible settings used by each search method during benchmark execution.

### `chart_of_accounts`
This block points to the accounting specification used by benchmark agents.

In practice, path resolution should normalize to the canonical file:

```text
configs/chart_of_accounts/chart_of_accounts.yaml
```

### `BASE_AGENTS`
This block defines the base agent templates used to populate each generated environment.

These templates typically specify:

- agent type,
- inventory strategy,
- firm-related goods,
- income statement type,
- chart-of-accounts path,
- price mapping.

---

## Configuration Notes

A few practical notes are important when working with the benchmark configuration.

### Graph suite and algorithm configuration are separate
The graph suite is maintained independently from the algorithm configuration. This separation is intentional and supports controlled reuse of graph families across experiment variants.

### Each graph can induce multiple environments
A single graph generally induces more than one environment, most notably:

- a standard environment, and
- one or more perturbed variants.

### Full benchmark burden scales multiplicatively
A full benchmark should be thought of as scaling approximately with:

```text
number_of_environments x number_of_algorithms x runs
```

This matters operationally, especially when the graph suite is large or the repetition schedule is increased.

### Population size is derived, not fixed globally
In `run_test_env.py`, population size is derived from:

```text
budget / generations
```

As a result, reducing the number of generations does not automatically guarantee a cheap run if the environment budget remains large.

### Use the canonical chart of accounts
The accounting configuration should resolve to:

```text
configs/chart_of_accounts/chart_of_accounts.yaml
```

This avoids ambiguity across local environments or scripts.

---

## Running Benchmark Experiments

## Minimal validation run

The following command provides a lightweight validation of the benchmark pipeline:

```bash
python src/experiment/run_experiment.py --algos generic --runs 1 --gens 5 --max-workers 1
```

### Interpretation
- `python src/experiment/run_experiment.py`: launches the benchmark orchestration script.
- `--algos generic`: restricts execution to the generic GA wrapper.
- `--runs 1`: executes a single repetition per generated environment.
- `--gens 5`: uses a very short evolutionary horizon for validation.
- `--max-workers 1`: disables parallel fan-out beyond one worker.

This run is useful as a first-pass system test before launching heavier benchmark suites.

---

## Full benchmark invocation

A representative full benchmark command is:

```bash
python src/experiment/run_experiment.py --algos flat,generic,joint,random,pso,macro_micro,macro,micro,recomb,no_crossover,mixed_generic --runs 6 --gens 100 --seed 42 --max-workers 4 --graphs configs/experiment_configs/input_graphs/test_suite_graphs.json --config configs/experiment_configs/algorithms_config/exp_config.json --chart configs/chart_of_accounts/chart_of_accounts.yaml --output ./exp_output --tag baseline
```

### Interpretation of principal arguments
- `--algos`: comma-separated subset of algorithms to execute.
- `--runs`: number of repetitions per algorithm and per environment.
- `--gens`: number of generations per run.
- `--seed`: base seed from which environment- and run-specific seeds are derived.
- `--max-workers`: number of worker processes used by the multiprocessing executor.
- `--graphs`: graph JSON file defining the benchmark DAG suite.
- `--config`: experiment configuration JSON.
- `--chart`: chart-of-accounts YAML file.
- `--output`: root output directory for timestamped result folders.
- `--tag`: optional identifier appended to the output-folder name.

---

## Benchmark Outputs

Each benchmark invocation creates a timestamped directory under the designated output root.

The principal artifacts are:

- `run_log.txt`
- `metadata.json`
- `config_used.json`
- `environment_database.csv`
- `results.csv`

### Typical roles of these files

- `run_log.txt`: execution trace and runtime information.
- `metadata.json`: run-level metadata describing the execution context.
- `config_used.json`: the effective configuration used in the run.
- `environment_database.csv`: environment-level metadata derived during environment generation.
- `results.csv`: benchmark results across algorithms, runs, and environments.

These files are the raw or semi-processed inputs for later experiment analysis.

---

## Reproducibility Notes for Benchmark Runs

- Benchmark runs should be launched from a terminal using the `.py` entry point.
- On Windows, multiprocessing is generally more reliable when the main process is started from a real script rather than from an interactive notebook or shell cell.
- A short validation run is strongly recommended before launching a full benchmark suite.
- The safest minimal path is to restrict execution to `--algos generic`.
- Runs that include `pso` should be launched from a writable working directory, because some dependencies may emit auxiliary files during execution.

---

# Part II. Fine-Tuning Pipeline

## Fine-Tuning Entry Point

The main script for fine-tuning execution is:

```text
src/fine_tuning/scripts/run_all.py
```

This pipeline is distinct from the benchmark workflow described above.

Its purpose is not to compare algorithms across a graph suite, but to evaluate candidate hyperparameter settings on a fixed environment.

---

## Fine-Tuning Inputs

The fine-tuning pipeline typically requires:

```text
configs/fine_tuning_configs/parameter_samples/plan.json
configs/fine_tuning_configs/parameter_samples/samples/
configs/chart_of_accounts/chart_of_accounts.yaml
```

### Role of each input

- `plan.json`: defines the environment, shared seed schedule, algorithms, budgets, and tuning bounds.
- `samples/`: contains candidate files such as `candidates_<algo>.json`.
- `chart_of_accounts.yaml`: accounting specification attached to the environment agents.

---

## Fine-Tuning Plan Structure

A representative plan has the following form:

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

---

## Interpretation of Fine-Tuning Fields

- `seeds`: repeated evaluation seeds used to assess robustness.
- `env_json`: environment specification for the fixed fine-tuning environment.
- `accounts_yaml`: chart-of-accounts file used by the environment agents.
- `algorithms`: list of algorithms included in the fine-tuning run.
- `generations`, `popsize`, `evals_cap`: shared computational budget parameters.
- `n_samples_per_algo`: nominal number of sampled candidates per algorithm.
- `bounds`: parameter ranges or grids defining the admissible tuning space for each algorithm.

---

## Candidate Files

Each algorithm typically requires a candidate file in the candidates directory, for example:

- `candidates_generic.json`
- `candidates_joint.json`
- `candidates_pso.json`

These files contain candidate identifiers and the corresponding hyperparameter dictionaries to be evaluated.

The fine-tuning pipeline combines those candidate definitions with the plan file and the evaluation seed schedule.

---

## Running Fine-Tuning

A representative command is:

```bash
python src/fine_tuning/scripts/run_all.py --plan configs/fine_tuning_configs/parameter_samples/plan.json --candidates-dir configs/fine_tuning_configs/parameter_samples/samples --out-base ./ft_output --algos generic,joint,pso --n-jobs 4 --emit-csv
```

### Interpretation of principal arguments
- `--plan`: fine-tuning plan JSON containing seeds, environment path, accounting path, and shared budgets.
- `--candidates-dir`: directory containing `candidates_<algo>.json` files.
- `--out-base`: base output directory for fine-tuning results.
- `--algos`: subset of algorithms to execute.
- `--n-jobs`: number of local worker processes.
- `--emit-csv`: requests per-algorithm CSV summaries in addition to per-run JSON outputs.

---

## Fine-Tuning Outputs

The fine-tuning pipeline writes:

- one JSON result per `(algorithm, candidate, seed)`,
- one directory per algorithm under the selected output base, and
- optional per-algorithm CSV summaries when `--emit-csv` is enabled.

These outputs are later inspected through the notebooks documented in:

```text
paper_results/README.md
```

---

## Conceptual Separation Between Benchmarking and Fine-Tuning

This distinction is important:

### Benchmarking
Benchmarking compares algorithms across many environments.

### Fine-tuning
Fine-tuning evaluates hyperparameter candidates on a fixed environment under repeated seeds.

The separation between `experiment` and `fine_tuning` is therefore methodological, not merely organizational.

---

# Part III. Connecting Execution with Paper Artifacts

## Where to inspect benchmark artifacts

The benchmark-result analysis notebook is documented in:

```text
paper_results/experiment/experiment_analysis.ipynb
```

## Where to inspect fine-tuning artifacts

The algorithm-specific tuning notebooks are organized under:

```text
paper_results/fine_tuning/individual_algorithm_results/
```

## Where to inspect the benchmark DAG documentation

The environment description materials are documented in:

```text
paper_results/docs/environment_description/
```

## Where to inspect the Ladder Strategy materials

The ladder-strategy notebook and PDF are documented in:

```text
paper_results/docs/ladder_strategy/
```

This is why execution documentation and artifact documentation are separated:
- this file explains how to run things;
- `paper_results/README.md` explains how to read and navigate what those runs produced.

---

# Part IV. Practical Guidance

## Recommended execution order for new users

A sensible workflow is:

1. verify the repository setup and dependencies;
2. launch a minimal benchmark validation run;
3. inspect the generated outputs;
4. launch a fuller benchmark only after the validation succeeds;
5. inspect benchmark-analysis materials in `paper_results/`;
6. run fine-tuning only when needed for parameter exploration or reproduction of tuning choices.

---

## Operational cautions

- Full benchmark runs can be expensive because environment count, algorithm count, and run count multiply quickly.
- A change in generations does not necessarily reduce cost proportionally when budgets drive population size.
- Different operating systems may handle multiprocessing differently.
- Fine-tuning should be treated as a separate experimental phase, not as a lightweight extension of the benchmark pipeline.

---

## Suggested minimal command set

### Validate benchmark pipeline
```bash
python src/experiment/run_experiment.py --algos generic --runs 1 --gens 5 --max-workers 1
```

### Launch full benchmark
```bash
python src/experiment/run_experiment.py --algos flat,generic,joint,random,pso,macro_micro,macro,micro,recomb,no_crossover,mixed_generic --runs 6 --gens 100 --seed 42 --max-workers 4 --graphs configs/experiment_configs/input_graphs/test_suite_graphs.json --config configs/experiment_configs/algorithms_config/exp_config.json --chart configs/chart_of_accounts/chart_of_accounts.yaml --output ./exp_output --tag baseline
```

### Launch fine-tuning
```bash
python src/fine_tuning/scripts/run_all.py --plan configs/fine_tuning_configs/parameter_samples/plan.json --candidates-dir configs/fine_tuning_configs/parameter_samples/samples --out-base ./ft_output --algos generic,joint,pso --n-jobs 4 --emit-csv
```

---

# Part V. Relation to Other Documentation

## Main project overview
For the project-level description of the framework, see:

```text
README.md
```

## Paper artifacts and interpretive materials
For the notebooks, PDFs, and research-facing outputs associated with the paper, see:

```text
paper_results/README.md
```

Together, the documentation is organized as follows:

- `README.md`: high-level project entry point;
- `docs/experiments.md`: operational execution guide;
- `paper_results/README.md`: paper-artifact navigation guide.

---

## Final Note

This document is intended to make the experimental layer of the repository inspectable, reproducible, and operationally clear. It consolidates the benchmark and fine-tuning workflows in a form suitable for technical users, collaborators, and readers interested in reproducing or extending the project.
