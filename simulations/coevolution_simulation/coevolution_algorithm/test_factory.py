

from simulations.coevolution_simulation.coevolution_algorithm.economy_factory import EconomyFactory

from simulations.coevolution_simulation.environment_builder.environment_builder import EnvironmentBuilder

from classes.economy.economy import Economy

from pathlib import Path



test_yaml_path = Path(__file__).resolve().parents[2] / "inputs" / "environments" / "3_node_linear_graph_env_1.yaml"

test_env_builder = EnvironmentBuilder()

test_env = test_env_builder.build_environment(test_yaml_path)


test_factory = EconomyFactory(test_env)

genome_evader = [0,1,1,0,1,0,1,0,1,0]

genome_auditor = [1,0,1,0,1,0,1,0,1,0]

economy = test_factory.create_economy(genome_evader, genome_auditor)

for order in economy.orders:
    print(order)    