from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown,
    nvmlDeviceGetTemperature, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
)
import time
import threading
import psutil
import os

from vllm import LLM, SamplingParams

################ GLOBAL CONSTANTS ################

# Set access token for HF.
os.environ["HF_TOKEN"] = ""

# To avoid any issues.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA devices that will be used for inference. For multiple devices: "0,1".
CUDA_VISIBLE_DEVICES = "0"
GPU_INDICES = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))

# Number of times that each experiment will be repeated.
ITERATIONS = 5
# Seconds for which server experiments will run for and timeout for batch experiments.
MAX_TEST_DURATION = 5 * 60
# For server experiments, we simulate a Poisson process, with λ set to queries per second (qps).
# Average monthly views per website 375773, top 0.5% websites have more than 10M monthly views.
# Source: https://blog.hubspot.com/website/web-traffic-analytics-report
# qps = monthly views / days / hours / min / secs
LAMBDA_QPS_ARRAY = [375773 / 30 / 24 / 60 / 60,
                    10000000 / 30 / 24 / 60 / 60]

LLM_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct", # max_model_len == 1024*10
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",      
    "google/gemma-2-9b-it",      
    #"google/gemma-3-1b-it",   
    "mistralai/Mistral-7B-Instruct-v0.3", # max_model_len == 1024*10
    "deepseek-ai/deepseek-llm-7b-chat"
]

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
    if model_name == "meta-llama/Llama-3.1-8B-Instruct" or model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        return LLM(model=model_name, dtype="auto", max_model_len=1024*10)
    else:
        return LLM(model=model_name, dtype="auto")
