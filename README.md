# ML Energy Benchmark
TODO: description

## Install
Clone repo with submodules (for EnergyMeter):
```
git clone https://github.com/maufadel/EnergyMeter.git
git submodule update --init --recursive
```
Run: `sudo apt-get install -y bpftrace` to install bpftrace, used by EnergyMeter to run.

## How to use
Simply run: `bash run_benchmark.sh`

## Note
Right now, the benchmark uses only the GPU corresponding to CUDA device 0. The monitoring also tracks only that device.
