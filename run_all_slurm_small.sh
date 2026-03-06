#!/bin/bash

# V100
sbatch --partition=gpu --nodelist=sanfrancisco --time=4320 ./run_benchmark_slurm.submit config-small.yaml

# A100
sbatch --partition=gpu --nodelist=sacramento --time=4320 ./run_benchmark_slurm.submit config-small.yaml

# H100
sbatch --partition=gpu_top --nodelist=sanjose --time=1440 ./run_benchmark_slurm.submit config-small.yaml

# H200
sbatch --partition=gpu_top --nodelist=trinity --time=1440 ./run_benchmark_slurm.submit config-small.yaml

# L40
sbatch --partition=gpu_top --nodelist=fresko --gres=gpu:l40spcie:1 --time=1440 ./run_benchmark_slurm.submit config-small.yaml
