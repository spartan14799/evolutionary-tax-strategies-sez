# Paper Results and Research Artifacts

This directory contains the principal documentary, analytical, and interpretive artifacts associated with the GECCO 2026 paper:

> **Evolutionary Computation for Tax-Minimizing Strategies in Special Economic Zones**

The purpose of this directory is not to host the executable core of the framework, but to preserve the materials that explain, inspect, and support the paper’s benchmark environments, experimental results, hyperparameter selection, and main strategic findings.

This README serves as a guided map to those materials.

---

## Purpose of This Directory

The contents of `paper_results/` are organized to support four complementary objectives:

- document the benchmark environments used in the experiments;
- preserve the analytical artifacts used to inspect benchmark outputs;
- record the fine-tuning and heuristic-selection process; and
- provide direct access to the materials used to interpret the paper’s central strategic result, namely the emergent **Ladder Strategy**.

In short, this directory is the documentary layer of the project: it contains the research artifacts that sit between the codebase and the final paper.

For the operational definition of algorithm parameters, benchmark configurations, fine-tuning inputs, execution commands, and output conventions, the repository-level reference is:

```text
docs/experiments.md
```

---

## Directory Structure

```bash
paper_results/
├── data_analysis/
├── docs/
├── experiment/
├── fine_tuning/
└── README.md
```

### Subdirectory roles
- `data_analysis/`: analytical inputs and configuration snapshots.
- `docs/`: explanatory and interpretive materials for benchmark environments and paper-level findings.
- `experiment/`: experiment-level outputs and post-processing artifacts related to the benchmark suite.
- `fine_tuning/`: artifacts related to hyperparameter exploration and parameter selection across algorithms.

---

## Recommended Reading Paths

Different readers usually come to this directory with different goals. The following paths are recommended.

### To understand the benchmark environments

Start with:

```text
paper_results/docs/environment_description/DAG_Structures.pdf
paper_results/docs/environment_description/Graphs_prop.ipynb
```

These materials explain the benchmark DAG families, their structural differences, and the rationale for using them as a documented environment suite rather than as an ad hoc set of graphs.

### To understand the paper’s main strategic result

Start with:

```text
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
paper_results/docs/ladder_strategy/ladder_strategy.pdf
```

These files provide the most direct reconstruction of the emergent Ladder Strategy reported in the paper.

### To inspect benchmark-result analysis

Start with:

```text
paper_results/experiment/experiment_analysis.ipynb
paper_results/data_analysis/experiment/configs/merged_config.json
```

These artifacts support the comparative inspection of benchmark outputs across algorithms and environments. The notebook provides the analytical narrative; the configuration snapshot preserves the merged benchmark specification used by the downstream analysis layer.

### To inspect fine-tuning and parameter selection

Start with:

```text
paper_results/fine_tuning/individual_algorithm_results/
```

This directory contains the algorithm-specific notebooks used to examine tuning outputs and select final heuristic configurations.

---

## Environment Documentation

The benchmark suite used in the paper is documented in:

```text
paper_results/docs/environment_description/DAG_Structures.pdf
```

This document is one of the foundational artifacts of the repository. It provides a structured description of the DAG families used in the experiments and clarifies the logic behind the benchmark design.

In particular, it documents:

- the production-graph families included in the benchmark suite;
- the structural differences between graph classes;
- the economic interpretation assigned to each graph family; and
- the relationship between graph structure, number of decisions, and search-space complexity.

The complementary notebook

```text
paper_results/docs/environment_description/Graphs_prop.ipynb
```

provides environment-level analysis and additional exploratory support for the benchmark design.

Taken together, these two artifacts define the benchmark suite as an explicitly analyzed and documented testbed.

---

## Ladder Strategy Materials

The paper’s main interpretive result is the emergence of a non-trivial profit-shifting pattern referred to as the **Ladder Strategy**.

The principal materials for that result are:

```text
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
paper_results/docs/ladder_strategy/ladder_strategy.pdf
```

### Roles of these files

- `ladder_strategy.ipynb`: executable and inspectable analytical notebook.
- `ladder_strategy.pdf`: stable reading artifact suitable for direct consultation alongside the paper.

These materials reconstruct how the strategy emerges under the modeled interaction between tax asymmetries and price asymmetries, and how it is interpreted within the broader institutional context of the paper.

They should be read together with the benchmark-environment documentation, since the strategy is inseparable from the structure of the production graph and the associated price system.

---

## Benchmark Experiment Analysis

The main notebook for inspecting and interpreting benchmark outputs is:

```text
paper_results/experiment/experiment_analysis.ipynb
```

This notebook documents the post-processing of benchmark runs and supports the comparative interpretation of experimental results reported in the paper.

Its role is to provide a research-facing view of the experiment outputs rather than to execute the experiments themselves.

Typical uses include:

- inspecting comparative algorithm performance;
- reviewing aggregate benchmark behavior across environments;
- supporting the interpretation of figures and tables reported in the paper; and
- checking the analytical consistency of reported findings.

The corresponding analysis-support artifacts are also preserved in:

```text
paper_results/data_analysis/
```

In particular, the repository currently includes:

```text
paper_results/data_analysis/experiment/configs/merged_config.json
```

This file preserves a merged configuration snapshot for the experiment-analysis layer.

The benchmark execution pipeline itself is documented separately in:

```text
docs/experiments.md
```

This separation is intentional: execution logic and research-artifact interpretation are documented in different places.

---

## Fine-Tuning Artifacts

The repository preserves the outputs and inspection notebooks used to analyze hyperparameter tuning and select the final heuristic configurations used in the paper.

These materials are organized under:

```text
paper_results/fine_tuning/individual_algorithm_results/
```

This directory contains algorithm-specific notebooks for reviewing tuning outcomes and identifying the final selected parameter settings.

The currently preserved notebooks include:

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

These notebooks collectively form the documentary record of how tuning outputs were inspected and how final heuristic configurations were selected for benchmark evaluation.

For the operational fine-tuning pipeline itself, including commands, plan files, candidate files, and output conventions, see:

```text
docs/experiments.md
```

---

## Price-System Interpretation

The benchmark price system is a central component of the modeled environments. While the code responsible for generating prices lives in the source tree, the interpretive use of those prices is reflected throughout the materials in this directory.

The two most relevant code paths are:

```text
src/experiment/exp_configuration.py
src/simulation/economy/order_book/utils/price_markup_generator.py
```

From the perspective of this directory:

- the environment-description materials explain the benchmark structures on which those prices operate;
- the experiment-analysis notebook inspects the downstream effect of those environments; and
- the ladder-strategy materials illustrate how specific price asymmetries interact with tax differentials to generate non-trivial strategies.

---

## How This Directory Relates to the Rest of the Repository

This directory should be read together with the following repository-level documentation:

### Main project overview

```text
README.md
```

This is the primary entry point to the repository and provides the high-level framing of the project.

### Experimental pipelines and reproducibility details

```text
docs/experiments.md
```

This document explains how to run benchmark experiments and fine-tuning workflows, how configurations are structured, and what outputs are generated.

In this sense:

- `README.md` explains the project,
- `docs/experiments.md` explains execution,
- `paper_results/README.md` explains the research artifacts and their analytical organization.

---

## Suggested Navigation by Reader Type

### For a reviewer or researcher interested in the benchmark design

Read in this order:

```text
paper_results/docs/environment_description/DAG_Structures.pdf
paper_results/docs/environment_description/Graphs_prop.ipynb
paper_results/experiment/experiment_analysis.ipynb
```

### For a reader interested in the paper’s main strategic finding

Read in this order:

```text
paper_results/docs/ladder_strategy/ladder_strategy.pdf
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
paper_results/docs/environment_description/DAG_Structures.pdf
```

### For a reader interested in heuristic tuning

Read in this order:

```text
paper_results/fine_tuning/individual_algorithm_results/
docs/experiments.md
```

### For a reader trying to connect code, benchmark structure, and interpretation

Read in this order:

```text
README.md
paper_results/docs/environment_description/DAG_Structures.pdf
src/experiment/exp_configuration.py
src/simulation/economy/order_book/utils/price_markup_generator.py
paper_results/data_analysis/experiment/configs/merged_config.json
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
paper_results/experiment/experiment_analysis.ipynb
```

---

## Notes on Scope

This directory is intended to preserve and organize the main paper-facing artifacts of the project. It does not replace:

- the source code in `src/`,
- the configuration files in `configs/`, or
- the execution documentation in `docs/experiments.md`.

Its role is narrower and more documentary: it helps the reader navigate the evidence, analyses, and interpretive materials associated with the paper.

---

## Summary of Key Files

### Benchmark-environment documentation

```text
paper_results/docs/environment_description/DAG_Structures.pdf
paper_results/docs/environment_description/Graphs_prop.ipynb
```

### Main strategic result

```text
paper_results/docs/ladder_strategy/ladder_strategy.ipynb
paper_results/docs/ladder_strategy/ladder_strategy.pdf
```

### Benchmark-result analysis

```text
paper_results/experiment/experiment_analysis.ipynb
paper_results/data_analysis/experiment/configs/merged_config.json
```

### Fine-tuning inspection materials

```text
paper_results/fine_tuning/individual_algorithm_results/
```

### Execution and reproducibility documentation

```text
docs/experiments.md
```

---

## Final Note

The materials collected here are meant to make the paper’s empirical and interpretive layer easier to inspect, navigate, and reproduce. They provide a structured bridge between the executable framework and the final scientific claims reported in the paper.
