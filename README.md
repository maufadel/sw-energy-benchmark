# ML Energy Benchmark
TODO: description

## Install
Clone repo with submodules:
```
git clone https://github.com/maufadel/sw-energy-benchmark.git
cd sw-energy-benchmark
git submodule update --init --recursive
```

EnergyMeter is used to track the energy used by the CPU, main memory, GPU and disk. The CPU and main memory are tracked through RAPL, so if RAPL is not available, only disk and GPU will be reported.


## How to use (bare metal)
1. Install `uv`:
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Set up Hugging Face token** (required):
   Create a `.env` file in the project root with your Hugging Face token:
   ```bash
   HF_TOKEN=your_token_here
   ```
   You can get a token from: https://huggingface.co/settings/tokens
   Alternatively, export it as an environment variable: `export HF_TOKEN=your_token`
3. Simply run: `./run_benchmark.sh config-v2.yaml 2>&1 | tee -a run_benchmark-config-v2-<GPU-NAME>.txt`
   The txt file will contain all the logging outputs. For a first run, it will be created. If you cancel the script in between and execute it again it will resume and continue to append to the file.

## How to use (SLURM cluster)

The SLURM workflow has two main components:

1. **`run_all_slurm.sh`** — orchestrator that splits a config into per-model batch configs and submits one SLURM job per model. Fully parameterized with `--partition`, `--nodelist`, `--time`, `--gres`, etc.
2. **`run_benchmark_slurm.submit`** — the compute-node submit script (called by the orchestrator): creates a venv, installs dependencies, downloads models, collects system info, and runs the benchmarks.

There is also an optional `run_benchmark_gpu.sh` convenience wrapper that maps a `--gpu` flag to cluster-specific SLURM parameters (partition, nodelist, gres, time). This is useful if you have a fixed cluster topology but is not required.

### Prerequisites
- Access to a SLURM cluster with GPU nodes
- `uv` installed (see Install section above)
- A Hugging Face token in a `.env` file at the project root (see bare metal instructions above)

### Basic usage
```bash
# Run all models from config-v2.yaml on a specific GPU node:
./run_all_slurm.sh --config config-v2.yaml --partition gpu --nodelist mynode --time 4320

# With account and specific GPU resource:
./run_all_slurm.sh --config config-v2.yaml --partition gpu_top --nodelist mynode --time 1440 \
  --account myaccount --gres gpu:1

# With GPU variant validation (retry on mismatch):
./run_all_slurm.sh --config config-v2.yaml --partition gpu --nodelist mynode --time 4320 \
  --expected-gpu H100_SXM --max-retries 3

# Dry run (show sbatch commands without submitting):
./run_all_slurm.sh --config config-v2.yaml --partition gpu --nodelist mynode --time 4320 --dry-run
```

### Resuming a failed run
If a run fails partway through, the temp config directory still contains unprocessed batch configs. Resume with:
```bash
./run_all_slurm.sh --resume temp_configs_config-v2 --partition gpu --nodelist mynode --time 4320
```

### Configurable environment variables
The submit script (`run_benchmark_slurm.submit`) supports these env vars to override default paths:

| Variable | Default | Description |
|---|---|---|
| `ROOT_FOLDER` | `/path/to/sw-energy-benchmark` | Project root on the cluster |
| `UV_PATH` | `$HOME/.local/bin/uv` | Path to the `uv` binary |
| `HF_HOME_DIR` | `/path/to/shared/huggingface-cache` | Shared Hugging Face cache directory |

## Note
Right now, the benchmark uses only the GPU corresponding to CUDA device 0. The monitoring also tracks only that device.
