

from Experiment_Pipeline.test_suite import generate_test_suite as gts
import numpy as np 
import os 

# Create RNG
rng = np.random.default_rng(42)


if __name__ == "__main__":
    # 👇 Get absolute path of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 👇 Build output folder path relative to this file
    output_dir = os.path.join(current_dir, "output_test_suites")

    output_path = gts.export_test_suite_to_json(
        output_dir=output_dir,       # ✅ ensures JSONs go inside test_suite/output_test_suites
        max_number_nodes=15,
        percentile=50,
        max_depth=8,
        k=10,
        start=10,
        step=0,
        n_agents=3,
        noise_cfg={"mean": 0, "std": 0.5},
        seed=42
    )

    print(f"✅ Test suite saved successfully to: {output_path}")
