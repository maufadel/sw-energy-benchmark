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


## How to use
1. Install `uv`:
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Set up Hugging Face token** (required):
   Create a `.env` file in the project root with your Hugging Face token:
   ```bash
   HF_TOKEN=your_token_here
   ```
   You can get a token from: https://huggingface.co/settings/tokens
   Alternatively, export it as an environment variable: `export HF_TOKEN=your_token`
3. Simply run: `./run_benchmark.sh config-150.yaml 2>&1 | tee -a run_benchmark-config-150-<GPU-NAME>.txt`
   The txt file will contain all the logging outputs. For a first run, it will be created. If you cancel the script in between and execute it again it will resume and continue to append to the file.

## Note
Right now, the benchmark uses only the GPU corresponding to CUDA device 0. The monitoring also tracks only that device.
