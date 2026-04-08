import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from pathlib import Path
import sys
import os

# =============================================================================
# Path a la raíz del proyecto (un nivel arriba del directorio actual)
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# IMPORTANTE: Asegúrate de que tu clase Economy sea importable aquí.
# =============================================================================
from src.simulation.economy.economy import Economy

# =============================================================================
# 1. Función de Aptitud (Fitness)
# =============================================================================
def calculate_fitness(genome, graph_links, price_matrix, agents_dict) -> float:
    """
    Calcula la aptitud de un genoma como la utilidad total (Net Profit agregado)
    reportada por la Economy (excluye MKT por diseño interno).
    """
    economy = Economy(
        production_graph=graph_links,
        pmatrix=price_matrix,
        agents_information=agents_dict,
        genome=genome.tolist() if isinstance(genome, np.ndarray) else list(genome)
    )
    results = economy.get_reports()
    total_utility = results.get('utility', 0.0)
    return float(total_utility)

# =============================================================================
# 2. Selección de Padres
# =============================================================================
def select_parents(population: np.ndarray, fitness: np.ndarray, num_parents: int) -> np.ndarray:
    """Selecciona los mejores individuos de la población para ser los padres."""
    parents = np.empty((num_parents, population.shape[1]))
    sorted_fitness_indices = np.argsort(fitness)[::-1]
    for i in range(num_parents):
        parents[i, :] = population[sorted_fitness_indices[i], :]
    return parents

# =============================================================================
# 3. Cruce (Crossover)
# =============================================================================
def crossover(parents: np.ndarray, offspring_size: Tuple[int, int]) -> np.ndarray:
    """Crea nuevos individuos (hijos) combinando los padres. Cruce de un solo punto (mitad)."""
    offspring = np.empty(offspring_size)
    crossover_point = offspring_size[1] // 2
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# =============================================================================
# 4. Mutación (binaria por flip)
# =============================================================================
def mutation(offspring_crossover: np.ndarray, mutation_rate: float = 0.05) -> np.ndarray:
    """Mutación para genomas binarios: con prob. 'mutation_rate' invierte el bit (0<->1)."""
    for idx in range(offspring_crossover.shape[0]):
        for gene_idx in range(offspring_crossover.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring_crossover[idx, gene_idx] = 1 - offspring_crossover[idx, gene_idx]
    return offspring_crossover

# =============================================================================
# Helper: asegurar restricción del último gen
# =============================================================================
def enforce_last_gene_zero(pop: np.ndarray) -> np.ndarray:
    """
    Fija el último gen en 0 para todos los individuos.
    Si tu restricción real es 1, cambia el 0 por 1.
    """
    pop[:, -1] = 1
    return pop

# =============================================================================
# Núcleo del Algoritmo Genético
# =============================================================================
def run_genetic_algorithm(
    graph_links,
    price_matrix,
    agents_dict,
    genome_shape: int,
    num_generations: int,
    population_size: int,
    num_parents_mating: int,
    mutation_rate: float = 0.05
):
    """Ejecuta el algoritmo genético con genomas binarios (0/1)."""

    # Población inicial BINARIA
    population = np.random.randint(0, 2, size=(population_size, genome_shape))
    enforce_last_gene_zero(population)

    best_fitness_per_generation: List[float] = []
    mean_fitness_per_generation: List[float] = []
    median_fitness_per_generation: List[float] = []

    print("--- Iniciando Algoritmo Genético ---")
    for generation in range(num_generations):
        # 1) Evaluación
        fitness = np.array([
            calculate_fitness(ind, graph_links, price_matrix, agents_dict)
            for ind in population
        ])

        best_f = float(np.max(fitness))
        mean_f = float(np.mean(fitness))
        median_f = float(np.median(fitness))

        best_fitness_per_generation.append(best_f)
        mean_fitness_per_generation.append(mean_f)
        median_fitness_per_generation.append(median_f)

        print(
            f"Generación {generation+1:03d}/{num_generations} | "
            f"Mejor: {best_f:.2f} | Media: {mean_f:.2f} | Mediana: {median_f:.2f}"
        )

        # 2) Selección
        parents = select_parents(population, fitness, num_parents_mating)

        # 3) Cruce
        offspring_crossover = crossover(
            parents,
            offspring_size=(population_size - parents.shape[0], population.shape[1])
        ).astype(int)
        enforce_last_gene_zero(offspring_crossover)

        # 4) Mutación
        offspring_mutation = mutation(offspring_crossover, mutation_rate)

        # Nueva generación (elitismo + hijos)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
        enforce_last_gene_zero(population)

    print("--- Algoritmo Genético Finalizado ---")

    # Mejor solución final
    final_fitness = np.array([
        calculate_fitness(ind, graph_links, price_matrix, agents_dict)
        for ind in population
    ])
    best_match_idx = int(np.argmax(final_fitness))
    best_solution_genome = population[best_match_idx, :]
    best_solution_fitness = float(final_fitness[best_match_idx])

    # Devolvemos historiales: mejor, media y mediana
    return (
        best_solution_genome,
        best_solution_fitness,
        best_fitness_per_generation,
        mean_fitness_per_generation,
        median_fitness_per_generation,
    )

# =============================================================================
# Parámetros del Algoritmo Genético
# =============================================================================
NUM_GENERATIONS = 15
POPULATION_SIZE = 30
NUM_PARENTS_MATING = 20
MUTATION_RATE = 0.5

# Genoma BINARIO de longitud 76
GENOME_SHAPE = 11

# =============================================================================
# --- Datos de entrada (los que ya tienes) ---
# =============================================================================
graph_links = [
  ("A", "B"),
  ("B", "C"),
  ("B", "D"),
  ("B", "E"),
  ("C", "E"),
  ("D", "E"),
  ("E", "F"),
]

# Deriva la lista de bienes a partir de los enlaces del grafo
goods = sorted({n for u, v in graph_links for n in (u, v)})

price_matrix_list = [
  [[7, 4, 8], [5, 7, 3], [7, 8, 5]],
  [[8, 5, 7], [7, 9, 3], [7, 8, 5]],
  [[10, 5, 11], [7, 9, 4], [10, 11, 8]],
  [[11, 7, 15], [7, 13, 4], [10, 17, 11]],
  [[18, 8, 15], [12, 15, 5], [16, 15, 13]],
  [[18, 12, 21], [14, 20, 7], [23, 25, 16]]
]
price_matrix = np.array(price_matrix_list, dtype=int)

accounts_path = ROOT_DIR / "chart_of_accounts.yaml"
agents_dict = {
    "MKT": {
        "type": "MKT",
        "inventory_strategy": "LIFO",
        "firm_related_goods": [],
        "income_statement_type": "standard",
        "accounts_yaml_path": accounts_path,
        "price_mapping": 0,
    },
    "NCT": {
        "type": "NCT",
        "inventory_strategy": "LIFO",
        "firm_related_goods": goods,
        "income_statement_type": "standard",
        "accounts_yaml_path": accounts_path,
        "price_mapping": 1,
    },
    "ZF": {
        "type": "ZF",
        "inventory_strategy": "LIFO",
        "firm_related_goods": goods,
        "income_statement_type": "standard",
        "accounts_yaml_path": accounts_path,
        "price_mapping": 2,
    },
}

# =============================================================================
# Ejecutar AG
# =============================================================================
best_genome, best_utility, best_history, mean_history, median_history = run_genetic_algorithm(
    graph_links=graph_links,
    price_matrix=price_matrix,
    agents_dict=agents_dict,
    genome_shape=GENOME_SHAPE,
    num_generations=NUM_GENERATIONS,
    population_size=POPULATION_SIZE,
    num_parents_mating=NUM_PARENTS_MATING,
    mutation_rate=MUTATION_RATE
)

print("\n--- Resultados ---")
print(f"Mejor utilidad total encontrada: {best_utility:.2f}")
print(f"Mejor genoma encontrado (binario): \n{best_genome}")

# =============================================================================
# Gráfica: Mejor vs Media vs Mediana
# =============================================================================
plt.figure()
plt.plot(best_history,   label="Mejor fitness")
plt.plot(mean_history,   label="Media del fitness")
plt.plot(median_history, label="Mediana del fitness")
plt.title("Evolución del Fitness por Generación")
plt.xlabel("Generación")
plt.ylabel("Fitness (Utilidad total)")
plt.grid(True)
plt.legend()
plt.show()
