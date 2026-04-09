import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_paths import get_default_chart_of_accounts_path, resolve_chart_of_accounts_path
from src.experiment.exp_configuration import full_environment_pipeline
from src.experiment.run_test_env import run_all_envs


# ---------------------------------------------------------------------------
# Helper: Export metadata and config snapshot
# ---------------------------------------------------------------------------
def export_experiment_metadata(out_dir: Path, config_path: Path, algos_used, runs, generations, seed):
    """Save configuration snapshot and metadata JSON to output folder."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "timestamp": timestamp,
        "algorithms_used": algos_used,
        "runs_per_algo": runs,
        "generations": generations,
        "base_seed": seed,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    with open(out_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)


# ---------------------------------------------------------------------------
# Helper: Dual stream logger (console + file)
# ---------------------------------------------------------------------------
class TeeLogger:
    """Redirects stdout to both console and a log file."""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment pipeline: build environments and execute algorithms."
    )

    # -----------------------------------------------------------------------
    # Add new algorithms to default CLI list
    # -----------------------------------------------------------------------
    parser.add_argument(
    "--algos",
    type=str,
    default="flat,generic,joint,random,pso,macro_micro,macro,micro,recomb,no_crossover,mixed_generic",
    help="Comma-separated list of algorithms to run (subset of config).",
    )
    parser.add_argument("--runs", type=int, default=6, help="Number of runs per algorithm.")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations per run.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output", type=str, default="./exp_output",
                        help="Output directory for results and metadata.")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum parallel workers for multiprocessing.")
    parser.add_argument(
        "--graphs",
        type=str,
        default="configs/experiment_configs/input_graphs/test_suite_graphs.json",
        help="Path to input graphs JSON.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_configs/algorithms_config/exp_config.json",
        help="Path to algorithm configuration JSON.",
    )
    parser.add_argument("--chart", type=str, default=str(get_default_chart_of_accounts_path()),
                        help="Path to chart_of_accounts.yaml.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag to name the experiment folder.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve paths and initialize output folder
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output)
    exp_name = f"experiment_{timestamp}" if not args.tag else f"experiment_{args.tag}_{timestamp}"
    exp_dir = output_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Start logging
    log_path = exp_dir / "run_log.txt"
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout
    print(f"Logging enabled at: {log_path}\n")

    # Paths
    charts_path = resolve_chart_of_accounts_path(args.chart)
    config_path = Path(args.config)
    graphs_path = Path(args.graphs)
    env_csv_path = exp_dir / "environment_database.csv"
    results_path = exp_dir / "results.csv"

    # -----------------------------------------------------------------------
    # Build Environments & Load Config
    # -----------------------------------------------------------------------
    print("Building environments and loading configurations...")
    algorithms, envs = full_environment_pipeline(
        graphs_json_path=graphs_path,
        config_json_path=config_path,
        output_csv=env_csv_path,
    )

    # -----------------------------------------------------------------------
    # Filter algorithms if subset specified
    # -----------------------------------------------------------------------
    selected_algos = [a.strip().lower() for a in args.algos.split(",")]
    algorithms = {k: v for k, v in algorithms.items() if k.lower() in selected_algos}

    if not algorithms:
        raise ValueError(f"No valid algorithms found in config matching {selected_algos}")

    print(f"Algorithms selected: {', '.join(algorithms.keys())}")

    # -----------------------------------------------------------------------
    # Export Metadata + Config Snapshot
    # -----------------------------------------------------------------------
    export_experiment_metadata(
        exp_dir,
        config_path,
        list(algorithms.keys()),
        args.runs,
        args.gens,
        args.seed,
    )

    # -----------------------------------------------------------------------
    # Execute Experiments
    # -----------------------------------------------------------------------
    print("Starting experiments...")
    run_all_envs(
        dicts_env=envs,
        configs=algorithms,
        chart_of_accounts_path=charts_path,
        runs=args.runs,
        generations=args.gens,
        path_output_db=results_path,
        max_workers=args.max_workers,
        base_seed=args.seed,
    )

    print(f"Experiment completed. Results saved in: {exp_dir}")

    # Restore stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
