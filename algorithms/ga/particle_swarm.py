import random
import time
import numpy as np
from pyswarms.discrete.binary import BinaryPSO
from typing import Any, Dict
from classes.economy.economy import Economy


import random
import time
import numpy as np
from pyswarms.discrete.binary import BinaryPSO
from typing import Any, Dict
from classes.economy.economy import Economy


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def _make_evaluator(production_graph, pmatrix, agents_information, fix_last_gene=True):
    """
    Returns a callable that evaluates one individual.

    If fix_last_gene=True, the last gene is forced to 1 before evaluation,
    ensuring that all evaluations correspond to genomes representing collusive behaviour

    This is Ensured Penalizing last position in NCT extremely by using np.inf
    """
    if fix_last_gene:

        def evaluate(individual: np.ndarray) -> float:
            """
            Evaluates the given individual. If the last gene is 0, assigns
            a utility of -np.inf. Otherwise, computes the utility using
            the Economy class.
            
            Parameters
            ----------
            individual : np.ndarray
                The individual to evaluate.
            
            Returns
            -------
            float
                The utility of the individual.
             """
            last_pos = individual[-1] 

            if last_pos == 0:
                u = -np.inf
            else:
                u = Economy(
                        production_graph=production_graph,
                        pmatrix=pmatrix,
                        agents_information=agents_information,
                        genome=list(individual),
                    ).get_reports().get("utility", 0.0)

            return float(u)
        
    else:
        def evaluate(individual: np.ndarray) -> float:
            u = Economy(
                production_graph=production_graph,
                pmatrix=pmatrix,
                agents_information=agents_information,
                genome=list(individual),
            ).get_reports().get("utility", 0.0)
            return float(u)

    return evaluate


# -----------------------------------------------------------------------------
# Objective Wrapper
# -----------------------------------------------------------------------------
def make_objective(production_graph, pmatrix, agents_information, fix_last_gene=True):
    """
    Wraps evaluator for PSO (minimization).
    PySwarms minimizes, so fitness is negated.
    """
    evaluate = _make_evaluator(production_graph, pmatrix, agents_information, fix_last_gene)

    def objective_function(X: np.ndarray) -> np.ndarray:
        # Apply transformation before evaluation
        fitness = np.array([evaluate(ind) for ind in X])
        return -fitness  # PSO minimizes
    return objective_function


# -----------------------------------------------------------------------------
# PSO Runner
# -----------------------------------------------------------------------------
def run_pso(
    production_graph,
    pmatrix,
    agents_information,
    genome_shape: int,
    generations: int = 15,
    c1: float = 0.3,
    c2: float = 0.3,
    w: float = 0.9,
    popsize: int = 30,
    fix_last_gene: bool = True,
    seed: int | None = 42,
    verbosity: int = 1,
    evals_cap: int | None = None,
    time_limit_sec: float | None = None,
    k: int = 3,
    p: int = 2
) -> Dict[str, Any]:
    """Runs a binary PSO and returns summary + curves.
    All evaluations, logs, and outputs are in terms of the *transformed* individuals.
    """
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.time()

    # Build objective
    objective_function = make_objective(production_graph, pmatrix, agents_information, fix_last_gene)

    # Initialize optimizer
    options = {"c1": c1, "c2": c2, "w": w, "k": k, "p": p}
    optimizer = BinaryPSO(
        n_particles=popsize,
        dimensions=int(genome_shape),
        options=options
    )

    # Run optimization
    cost, pos = optimizer.optimize(
        objective_function,
        iters=generations,
        verbose=(verbosity > 0)
    )
    print("cost_history length:", len(optimizer.cost_history))
    print("mean_pbest_history length:",
      len(getattr(optimizer, "mean_pbest_history", [])))


    runtime = time.time() - t0



    # Extract curves (invert back to utilities)
    best_curve = -np.array(optimizer.cost_history)
    mean_curve = -np.array(getattr(optimizer, "mean_pbest_history", []))

    best_utility = float(np.max(best_curve))
    best_genome = list(map(int, pos))

    # Optional debugging prints for verification
    if verbosity > 0:
        print("\n=== FINAL RESULTS ===")
        print(f"Best genome: {best_genome}")
        print(f"Best utility: {best_utility:.4f}")
        print(f"Runtime: {runtime:.2f} sec")
        print("========================================\n")

    return {
        "best_genome": best_genome,
        "best_utility": best_utility,
        "curves": {
            "best": best_curve.tolist(),
            "mean": mean_curve.tolist() if len(mean_curve) > 0 else [],
        },
        "meta": {
            "genome_shape": int(genome_shape),
            "generations": int(generations),
            "popsize": int(popsize),
            "fix_last_gene": bool(fix_last_gene),
            "seed": seed,
            "runtime_sec": float(runtime),
            "evals_total": int(popsize * generations),
            "budget": {
                "evals_cap": evals_cap,
                "time_limit_sec": time_limit_sec,
                "triggered": False,
                "reason": None,
            },
        },
    }
