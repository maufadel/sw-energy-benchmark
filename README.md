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
2. Simply run: `bash run_benchmark.sh`

## Note
Right now, the benchmark uses only the GPU corresponding to CUDA device 0. The monitoring also tracks only that device.
