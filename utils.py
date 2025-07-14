from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown,
    nvmlDeviceGetTemperature, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
)
import time
import threading
import psutil
import os
os.environ["HF_HUB_XET_DISABLED"] = "1"
import yaml
from dotenv import load_dotenv

from huggingface_hub import snapshot_download, HfFolder
from dotenv import load_dotenv

import torch
from vllm import LLM, SamplingParams


################ GLOBAL CONSTANTS ################

config = None
CUDA_VISIBLE_DEVICES = None
GPU_INDICES = None
ITERATIONS = None
MAX_TEST_DURATION = None
LAMBDA_QPS_ARRAY = None
LLM_MODELS = None

def load_config(config_path='config.yaml'):
    global config, CUDA_VISIBLE_DEVICES, GPU_INDICES, ITERATIONS, MAX_TEST_DURATION, LAMBDA_QPS_ARRAY, LLM_MODELS
    with open(config_path, "r") as f:
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


def download_model(model_name: str):
    """
    Downloads a single model from the Hugging Face Hub to the local cache.
    It handles loading the HF_TOKEN from a .env file internally.

    Args:
        model_name (str): The model repository ID to download.
    """
    # --- Load Hugging Face Token ---
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("Warning: HF_TOKEN not found. This may fail for gated models.")
    else:
        # This only needs to be done once, but it's safe to call multiple times.
        HfFolder.save_token(hf_token)

    print("="*50)
    print(f"Attempting to download model: {model_name}")
    print("="*50)
    try:
        # snapshot_download downloads the entire repository to the cache.
        # It's idempotent, meaning it won't re-download files that already exist.
        snapshot_download(
            repo_id=model_name,
            token=hf_token,  # Pass the token for gated models
            local_dir_use_symlinks=False # Recommended for stability
        )
        print(f"--> SUCCESS: Model '{model_name}' is available in local cache.\n")
    except Exception as e:
        print(f"--> ERROR: Failed to download model '{model_name}'.")
        print(f"    Reason: {e}\n")

def create_vllm(model_name):
    dtype = "auto"  # Default dtype
    device_name = "CPU"

    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        # Set dtype based on the detected GPU name
        if any(gpu in device_name for gpu in ["V100", "T4"]):
            dtype = "float16"
        elif any(gpu in device_name for gpu in ["H100", "H200", "A100", "L4", "A30"]):
            dtype = "auto"
        # For other GPUs, the default 'auto' is generally a safe choice.
    print(f"Detected device: {device_name}.")
    print(f"Using dtype '{dtype}' for model '{model_name}'.")
    
    try:
        print("Attempting to load model with default settings...")
        llm = LLM(
                model=model_name,
                dtype=dtype,
                trust_remote_code=True
                )
        return llm
    except RuntimeError as e:
        error_message = str(e)
        if "increase `gpu_memory_utilization`" in error_message and "decreasing `max_model_len`" in error_message:
            print("Default model loading failed due to memory constraints.")
            print("Retrying with memory optimization settings...")
            llm = LLM(
                model=model_name,
                dtype=dtype,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                max_model_len=16384
            )
            return llm
        else:
            # Re-raise the exception if it's not the one we can handle
            raise e

def create_vllm_old(model_name):
    #download_model(model_name)
    dtype = "float16" # float16 for v100, auto for H100, H200, A100
    print(f"Using dtype '{dtype}' for model {model_name}.")
    return LLM(model=model_name, dtype=dtype, trust_remote_code=True)
    if "mistral" in model_name:
        print("Using Mistral tokenizer")
        return LLM(model=model_name, dtype=dtype, trust_remote_code=True, tokenizer_mode="mistral")
    
    if model_name == "meta-llama/Llama-3.1-8B-Instruct" or model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        return LLM(model=model_name, dtype=dtype, max_model_len=1024*10)
    else:
        return LLM(model=model_name, dtype=dtype, trust_remote_code=True)
    #elif model_name == "google/gemma-2-2b-it.":
    #    return LLM(model=model_name, dtype="bfloat16")
