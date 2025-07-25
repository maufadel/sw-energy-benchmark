from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown,
    nvmlDeviceGetTemperature, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
)
import time
import threading
import psutil
import os
import yaml
from dotenv import load_dotenv

import torch
from vllm import LLM, SamplingParams

################ GLOBAL CONSTANTS ################

# Load config from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set access token for HF.
load_dotenv()

# To avoid any issues.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA devices that will be used for inference. For multiple devices: "0,1".
CUDA_VISIBLE_DEVICES = config["CUDA_VISIBLE_DEVICES"]
GPU_INDICES = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))

# Number of times that each experiment will be repeated.
ITERATIONS = config["ITERATIONS"]
# Seconds for which server experiments will run for and timeout for batch experiments.
MAX_TEST_DURATION = config["MAX_TEST_DURATION"]
# For server experiments, we simulate a Poisson process, with λ set to queries per second (qps).
LAMBDA_QPS_ARRAY = config["LAMBDA_QPS_ARRAY"]

LLM_MODELS = config["LLM_MODELS"]

################### CLASSES #####################

class MonitorThread(threading.Thread):
    def __init__(self, gpu_indices=None, secs_between_samples=1):
        super().__init__(daemon=True)
        if gpu_indices:
            self.gpu_indices = gpu_indices
        else:
            # Monitor all GPUs.
            self.gpu_indices = GPU_INDICES
        self.secs_between_samples = secs_between_samples
        self.running = True

        self.gpu_handles = {i: nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices}

        self.gpu_utilization = {i: [] for i in self.gpu_indices}
        self.gpu_mem_utilization = {i: [] for i in self.gpu_indices}
        self.gpu_power_draw = {i: [] for i in self.gpu_indices}

        self.cpu_utilization = []
        self.ram_utilization = []

    def run(self):
        while self.running:
            for i in self.gpu_indices:
                handle = self.gpu_handles[i]
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle)

                self.gpu_utilization[i].append(util.gpu)
                self.gpu_mem_utilization[i].append(mem.used / (1024 * 1024))  # in MB
                self.gpu_power_draw[i].append(power / 1000.0)  # convert to Watts

            self.ram_utilization.append(psutil.virtual_memory().used / (1024 * 1024))
            self.cpu_utilization.append(psutil.cpu_percent())

            time.sleep(self.secs_between_samples)

    def stop(self):
        self.running = False
        self.join()

    def get_all_metrics(self):
        all_metrics = {}

        for i in self.gpu_indices:
            all_metrics[f"gpu_{i}_memory_used_mb"] = self.gpu_mem_utilization[i]
            all_metrics[f"gpu_{i}_utilization_percent"] = self.gpu_utilization[i]
            all_metrics[f"gpu_{i}_power_draw_watts"] = self.gpu_power_draw[i]

        all_metrics["cpu_memory_used_mb"] = self.ram_utilization
        all_metrics["cpu_utilization_percent"] = self.cpu_utilization

        return all_metrics

################ STATIC METHODS ################

def wait_for_gpu_cooldown(gpu_handle, target_temp=55, check_interval=5):
    """Wait until the GPU temperature drops below the target temperature."""
    gpu_temp = nvmlDeviceGetTemperature(gpu_handle, 0)
    print(f"GPU temperature before cooldown: {gpu_temp}°C")

    while gpu_temp > target_temp:
        print(f"Waiting for GPU to cool down... Current temp: {gpu_temp}°C")
        time.sleep(check_interval)
        gpu_temp = nvmlDeviceGetTemperature(gpu_handle, 0)

    print(f"GPU cooled down to {gpu_temp}°C. Continue.")

def create_vllm(model_name):
    dtype = "auto"
    print(f"Using dtype '{dtype}' for model {model_name}.")
    
    try:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct" or model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            return LLM(model=model_name, dtype=dtype, max_model_len=1024*10)
        else:
            return LLM(model=model_name, dtype=dtype)
    except Exception as e:
        print(f"Error loading model {model_name} with dtype '{dtype}' - exception: {e}. Will try loading with float16.")
        if model_name == "meta-llama/Llama-3.1-8B-Instruct" or model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            return LLM(model=model_name, dtype="float16", max_model_len=1024*10)
        else:
            return LLM(model=model_name, dtype="float16")
