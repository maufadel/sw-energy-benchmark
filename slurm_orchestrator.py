
import yaml
import sys
import os
import math

def create_batch_configs(main_config_path, batch_size=10):
    """
    Reads a large YAML config file, splits the 'LLM_MODELS' list into smaller
    batches, and creates a temporary config file for each batch.
    """
    if not os.path.exists(main_config_path):
        print(f"Error: Main config file not found at {main_config_path}")
        sys.exit(1)

    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)

    all_models = main_config.get("LLM_MODELS", [])
    if not all_models:
        print("Warning: No 'LLM_MODELS' found in the config file.")
        return

    num_batches = math.ceil(len(all_models) / batch_size)
    print(f"Splitting {len(all_models)} models into {num_batches} batches of (up to) {batch_size} models each.")

    # Create a directory for temporary configs
    temp_config_dir = "temp_configs"
    os.makedirs(temp_config_dir, exist_ok=True)
    print(f"Temporary config files will be stored in: {temp_config_dir}")


    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        model_batch = all_models[batch_start:batch_end]

        batch_config = main_config.copy()
        batch_config["LLM_MODELS"] = model_batch

        batch_config_filename = os.path.join(temp_config_dir, f"config_batch_{i+1}.yaml")
        with open(batch_config_filename, 'w') as f:
            yaml.dump(batch_config, f, default_flow_style=False)
        
        print(f"  -> Created {batch_config_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slurm_orchestrator.py <path_to_main_config> [batch_size]")
        sys.exit(1)
    
    main_config_file = sys.argv[1]
    size = 10
    if len(sys.argv) > 2:
        try:
            size = int(sys.argv[2])
        except ValueError:
            print("Error: batch_size must be an integer.")
            sys.exit(1)

    create_batch_configs(main_config_file, size)
