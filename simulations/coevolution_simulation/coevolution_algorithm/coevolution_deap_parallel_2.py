

### Author: ANDRES LEGUIZAMON / CARLOS SANCHEZ / GERMAN RODRIGUEZ / SOFIA OCAMPO 
### 2025
### TODO : LICENSE / COPYRIGHT 



import time

from functools import partial

import random
import multiprocessing
from datetime import datetime
from pathlib import Path
import pandas as pd
from deap import base, creator, tools , algorithms

from simulations.coevolution_simulation.coevolution_algorithm.economy_factory import EconomyFactory

from simulations.coevolution_simulation.environment_builder.environment_builder import EnvironmentBuilder

from classes.economy.economy import Economy

import copy


# ==============================================================
# 1. GLOBAL FUNCTION – must be at module level for multiprocessing
# ==============================================================

def evaluate_pair(args, function_type="standard"):
    """
    Evaluate one pair of evader and auditor genomes.
    Executed in parallel via multiprocessing.

    Parameters
    ----------
    args : tuple
        (genome_evader, genome_auditor, factory)
    function_type : str, optional
        Determines which coevolution function is used.
        Options:
        - 'standard': uses GA_coevolution_function
        - 'alternative': uses GA_coevolution_function_ev_neutral_aud_neutral

    Returns
    -------
    tuple
        (fitness_evader, fitness_auditor)
    """
    genome_e, genome_a, factory = args
    economy = factory.create_economy(genome_e, genome_a)

    if function_type == "standard":
        fit_e, fit_a = economy.GA_coevolution_function_1()
    elif function_type == "alternative":
        fit_e, fit_a = economy.GA_coevolution_function_ev_neutral_aud_neutral()
    else:
        raise ValueError(f"Unknown function_type: {function_type}")

    return fit_e, fit_a


def run_coevolution_parallel(
    yaml_path: Path,
    binary_length: int = 20,
    int_length: int = 20,
    int_k: int = 25,
    n_generations: int = 200,
    pop_size: int = 20,
    function_type: str = "standard",
    opponent_sample_proportion: float = 0.5,
    cx_prob: float = 0.7,
    mut_prob: float = 0.6,
    output_dir: Path = Path("output"),
):
    """
    Run a coevolutionary DEAP simulation between two adversarial species
    (evaders and auditors) using multiprocessing for parallel evaluation.
    """
    start_time = time.time()

    # === Environment Setup ===
    env_builder = EnvironmentBuilder()
    test_env = env_builder.build_environment(yaml_path)
    test_factory = EconomyFactory(test_env)

    # Opponent samples (at least 1, at most pop_size)
    opponent_samples = max(1, int(round(pop_size * opponent_sample_proportion)))
    opponent_samples = min(opponent_samples, pop_size)

    # === DEAP Setup (guard creator types for re-runs) ===
    if not hasattr(creator, "FitnessEvader"):
        creator.create("FitnessEvader", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualEvader"):
        creator.create("IndividualEvader", list, fitness=creator.FitnessEvader)

    if not hasattr(creator, "FitnessAuditor"):
        creator.create("FitnessAuditor", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualAuditor"):
        creator.create("IndividualAuditor", list, fitness=creator.FitnessAuditor)

    toolbox_e = base.Toolbox()
    toolbox_a = base.Toolbox()

    # Evader (binary) builders
    toolbox_e.register("attr_bool", random.randint, 0, 1)
    toolbox_e.register("individual", tools.initRepeat, creator.IndividualEvader, toolbox_e.attr_bool, binary_length)
    toolbox_e.register("population", tools.initRepeat, list, toolbox_e.individual)

    # Auditor (integer) builders
    toolbox_a.register("attr_int", random.randint, 0, int_k)
    toolbox_a.register("individual", tools.initRepeat, creator.IndividualAuditor, toolbox_a.attr_int, int_length)
    toolbox_a.register("population", tools.initRepeat, list, toolbox_a.individual)

    # Genetic operators
    toolbox_e.register("mate", tools.cxTwoPoint)
    toolbox_e.register("mutate", tools.mutFlipBit, indpb=1.0 / max(1, binary_length))
    toolbox_e.register("select", tools.selTournament, tournsize=3)
    toolbox_e.register("clone", copy.deepcopy)

    toolbox_a.register("mate", tools.cxTwoPoint)
    toolbox_a.register("mutate", tools.mutUniformInt, low=0, up=int_k, indpb=1.0 / max(1, int_length))
    toolbox_a.register("select", tools.selTournament, tournsize=3)
    toolbox_a.register("clone", copy.deepcopy)

    # Initial populations
    pop_e = toolbox_e.population(n=pop_size)
    pop_a = toolbox_a.population(n=pop_size)

    # === Multiprocessing Setup ===
    n_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_cores)

    evaluate_func = partial(evaluate_pair, function_type=function_type)
    toolbox_e.register("map", pool.map)
    toolbox_a.register("map", pool.map)

    print(f"Starting PARALLEL coevolution ({n_cores} cores)")
    print(f"Generations: {n_generations} | Population: {pop_size} | Opponents per ind.: {opponent_samples}")

    # === Results DataFrame ===
    df_results = pd.DataFrame(
        columns=[
            "generation",
            "best_fitness_evader",
            "best_fitness_auditor",
            "mean_fitness_evader",
            "std_fitness_evader",
            "mean_fitness_auditor",
            "std_fitness_auditor",
            "best_evader_genome",
            "best_auditor_genome",
        ]
    )

    # === Evaluation helper (averages across sampled opponents) ===
    def evaluate_population(pop_e_in, pop_a_in):
        # Evaders vs sampled auditors
        pairs_e = []
        for e in pop_e_in:
            opponents = random.sample(pop_a_in, k=opponent_samples)
            pairs_e.extend((e, a, test_factory) for a in opponents)
        res_e = toolbox_e.map(evaluate_func, pairs_e)
        for i, e in enumerate(pop_e_in):
            start = i * opponent_samples
            chunk = res_e[start : start + opponent_samples]
            e.fitness.values = (sum(r[0] for r in chunk) / opponent_samples,)

        # Auditors vs sampled evaders
        pairs_a = []
        for a in pop_a_in:
            opponents = random.sample(pop_e_in, k=opponent_samples)
            pairs_a.extend((e, a, test_factory) for e in opponents)
        res_a = toolbox_a.map(evaluate_func, pairs_a)
        for i, a in enumerate(pop_a_in):
            start = i * opponent_samples
            chunk = res_a[start : start + opponent_samples]
            a.fitness.values = (sum(r[1] for r in chunk) / opponent_samples,)

    try:
        # === Evolution loop ===
        for gen in range(n_generations):
            # 1) Variation
            off_e = list(map(toolbox_e.clone, pop_e))
            off_a = list(map(toolbox_a.clone, pop_a))
            off_e = algorithms.varAnd(off_e, toolbox_e, cx_prob, mut_prob)
            off_a = algorithms.varAnd(off_a, toolbox_a, cx_prob, mut_prob)

            # 2) Invalidate fitness after variation  (IMPORTANT: use 'del', not empty tuple)
            for ind in off_e:
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values
            for ind in off_a:
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values

            # 3) Evaluate offspring against current opposite population
            evaluate_population(off_e, pop_a)
            evaluate_population(pop_e, off_a)

            # 4) Selection (μ + λ)
            pop_e = toolbox_e.select(pop_e + off_e, k=pop_size)
            pop_a = toolbox_a.select(pop_a + off_a, k=pop_size)

            # 5) Logging
            fits_e = [ind.fitness.values[0] for ind in pop_e]
            fits_a = [ind.fitness.values[0] for ind in pop_a]
            best_e = tools.selBest(pop_e, 1)[0]
            best_a = tools.selBest(pop_a, 1)[0]

            df_results.loc[len(df_results)] = [
                gen,
                max(fits_e),
                max(fits_a),
                sum(fits_e) / len(fits_e),
                pd.Series(fits_e).std(ddof=0),
                sum(fits_a) / len(fits_a),
                pd.Series(fits_a).std(ddof=0),
                list(best_e),
                list(best_a),
            ]

            print(
                f"Gen {gen+1}/{n_generations} | "
                f"Evader best: {max(fits_e):.4f} | "
                f"Auditor best: {max(fits_a):.4f}"
            )
    finally:
        pool.close()
        pool.join()

    # === Runtime ===
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Parallel coevolution finished in {int(mins)}m {int(secs)}s")

    # === Save results ===
    output_dir.mkdir(exist_ok=True)
    yaml_name = Path(yaml_path).stem
    date_str = datetime.now().strftime("%Y%m%d")
    csv_filename = f"coevolution_results_{yaml_name}_{date_str}_{function_type}.csv"
    csv_path = output_dir / csv_filename
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return {
        "results_df": df_results,
        "csv_path": csv_path,
        "final_pop_evaders": pop_e,
        "final_pop_auditors": pop_a,
    }


if __name__ == "__main__":
    test_yaml_path = (
        Path(__file__).resolve().parents[2] / "inputs" / "environments" / "6_node_nested_graph_env_1.yaml"
    )
    df_results = run_coevolution_parallel(test_yaml_path, function_type='standard')