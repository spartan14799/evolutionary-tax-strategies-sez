if __name__ == "__main__":
    test_yaml_path = (
        Path(__file__).resolve().parents[2] / "inputs" / "environments" / "3_node_linear_graph_env_1.yaml"
    )
    df_results = run_coevolution(test_yaml_path)