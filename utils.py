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


def cleanup_gpu_memory(verbose=True):
    """
    Aggressive GPU memory cleanup before loading a new model.
    This ensures the GPU is in a clean state by:
    1. Killing any orphaned Ray processes
    2. Clearing PyTorch cache
    3. Running garbage collection
    4. Resetting CUDA context if possible

    Args:
        verbose: Whether to print detailed cleanup messages

    Returns:
        bool: True if cleanup completed without errors
    """
    import gc
    import subprocess

    cleanup_success = True

    if verbose:
        print(f"\n{'='*60}")
        print("Pre-model GPU memory cleanup")
        print(f"{'='*60}")

    # Step 1: Kill any orphaned Ray processes
    try:
        if verbose:
            print("[1/5] Checking for orphaned Ray processes...")
        result = subprocess.run(
            "pgrep -f 'ray::' || true",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            ray_pids = result.stdout.strip().split('\n')
            if verbose:
                print(f"      Found {len(ray_pids)} orphaned Ray processes, killing them...")
            for pid in ray_pids:
                if pid.strip():
                    subprocess.run(f"kill -9 {pid.strip()}", shell=True, timeout=2)
            if verbose:
                print("      ✓ Killed orphaned Ray processes")
        else:
            if verbose:
                print("      ✓ No orphaned Ray processes found")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ Ray process cleanup failed: {e}")

    # Step 2: Clear PyTorch CUDA cache
    try:
        if verbose:
            print("[2/5] Clearing PyTorch CUDA cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if verbose:
                print("      ✓ CUDA cache cleared")
        else:
            if verbose:
                print("      ⓘ CUDA not available (skipped)")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ CUDA cache clear failed: {e}")

    # Step 3: Run garbage collection
    try:
        if verbose:
            print("[3/5] Running Python garbage collection...")
        gc.collect()
        if verbose:
            print("      ✓ Garbage collection complete")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ Garbage collection failed: {e}")

    # Step 4: Synchronize CUDA to ensure all operations complete
    try:
        if verbose:
            print("[4/5] Synchronizing CUDA operations...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if verbose:
                print("      ✓ CUDA synchronized")
        else:
            if verbose:
                print("      ⓘ CUDA not available (skipped)")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ CUDA sync failed: {e}")

    # Step 5: Report GPU memory status
    try:
        if verbose:
            print("[5/5] Checking GPU memory status...")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            if verbose:
                print(f"      GPU memory allocated: {allocated:.2f} GB")
                print(f"      GPU memory reserved: {reserved:.2f} GB")
                if allocated < 0.1:
                    print("      ✓ GPU memory is clean")
                else:
                    print(f"      ⚠ Warning: {allocated:.2f} GB still allocated")
        else:
            if verbose:
                print("      ⓘ CUDA not available (skipped)")
    except Exception as e:
        if verbose:
            print(f"      ⓘ Could not check GPU memory: {e}")

    if verbose:
        print(f"{'='*60}")
        if cleanup_success:
            print("✓ Pre-model cleanup completed successfully")
        else:
            print("⚠ Pre-model cleanup completed with warnings")
        print(f"{'='*60}\n")

    return cleanup_success


def cleanup_vllm_resources(llm=None, model_name="unknown", verbose=True):
    """
    Comprehensive cleanup of vLLM, Ray, and CUDA resources.

    This function implements a multi-step cleanup process to handle the known
    issues with vLLM where Ray workers and CUDA contexts persist after
    exceptions. This is particularly important when models fail to load or
    encounter errors during inference.

    Args:
        llm: The vLLM LLM object to clean up (can be None)
        model_name: Name of the model for logging purposes
        verbose: Whether to print detailed cleanup messages

    Returns:
        bool: True if cleanup completed without errors, False otherwise
    """
    import gc
    import traceback

    cleanup_success = True

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting cleanup for model: {model_name}")
        print(f"{'='*60}")

    # Step 1: Synchronize CUDA operations
    try:
        if torch.cuda.is_available():
            if verbose:
                print("[1/7] Synchronizing CUDA operations...")
            torch.cuda.synchronize()
            if verbose:
                print("      ✓ CUDA synchronized")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ CUDA sync failed: {e}")

    # Step 2: Delete the LLM object
    try:
        if llm is not None:
            if verbose:
                print("[2/7] Deleting vLLM LLM object...")
            del llm
            if verbose:
                print("      ✓ LLM object deleted")
        else:
            if verbose:
                print("[2/7] No LLM object to delete (skipped)")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ LLM deletion failed: {e}")

    # Step 3: Attempt to destroy model parallel groups
    try:
        if verbose:
            print("[3/7] Destroying model parallel groups...")
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if verbose:
                print("      ✓ Model parallel groups destroyed")
        except ImportError:
            if verbose:
                print("      ⓘ destroy_model_parallel not available (older vLLM version)")
        except Exception as inner_e:
            if verbose:
                print(f"      ⚠ Model parallel cleanup warning: {inner_e}")
    except Exception as e:
        # Don't mark as failure - this is optional cleanup
        if verbose:
            print(f"      ⓘ Model parallel cleanup skipped: {e}")

    # Step 4: Destroy PyTorch distributed process group
    try:
        if verbose:
            print("[4/7] Destroying distributed process group...")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            if verbose:
                print("      ✓ Process group destroyed")
        else:
            if verbose:
                print("      ⓘ No distributed process group initialized (skipped)")
    except Exception as e:
        # Don't mark as failure - process group may not exist
        if verbose:
            print(f"      ⓘ Process group cleanup skipped: {e}")

    # Step 5: Shutdown Ray if it's running
    try:
        if verbose:
            print("[5/7] Shutting down Ray workers...")
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                if verbose:
                    print("      ✓ Ray shutdown complete")
            else:
                if verbose:
                    print("      ⓘ Ray not initialized (skipped)")
        except ImportError:
            if verbose:
                print("      ⓘ Ray not imported (skipped)")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ Ray shutdown failed: {e}")

    # Step 6: Clear CUDA cache and run garbage collection
    try:
        if verbose:
            print("[6/7] Clearing CUDA cache and running garbage collection...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            print("      ✓ Cache cleared and garbage collected")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ Cache cleanup failed: {e}")

    # Step 7: Final CUDA synchronization
    try:
        if torch.cuda.is_available():
            if verbose:
                print("[7/7] Final CUDA synchronization...")
            torch.cuda.synchronize()
            if verbose:
                print("      ✓ Final CUDA sync complete")
    except Exception as e:
        cleanup_success = False
        if verbose:
            print(f"      ✗ Final CUDA sync failed: {e}")

    if verbose:
        print(f"{'='*60}")
        if cleanup_success:
            print(f"✓ Cleanup completed successfully for: {model_name}")
        else:
            print(f"⚠ Cleanup completed with warnings for: {model_name}")
        print(f"{'='*60}\n")

    return cleanup_success


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
            try:
                llm = LLM(
                    model=model_name,
                    dtype=dtype,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.95,
                    max_model_len=16384
                )
                print("Succesfully created th model with memory optimization settings")
                return llm
            except:
                print("Could not run model even with memory optimization settings")
                raise e
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
