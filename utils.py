from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTemperature, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage,
    nvmlDeviceGetClockInfo, NVML_CLOCK_SM
)
import time
import threading
import psutil
import os
import random
import numpy as np
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
SAMPLING_PARAMS = None
WARMUP_DURATION = None
CONTEXT_LENGTH = None


def load_config(config_path='config.yaml'):
    global config, CUDA_VISIBLE_DEVICES, GPU_INDICES, ITERATIONS, MAX_TEST_DURATION, LAMBDA_QPS_ARRAY, LLM_MODELS, CONTEXT_LENGTH, WARMUP_DURATION
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
    # max_model_len parameter of vLLM that defines the maximum length of the input.
    CONTEXT_LENGTH = config["CONTEXT_LENGTH"]
    # Warmup time for server scenario. For batch, the warmup is one batch.
    WARMUP_DURATION = config["WARMUP_DURATION"]
    # Seconds for which server experiments will run for and timeout for batch experiments.
    MAX_TEST_DURATION = config["MAX_TEST_DURATION"]
    # For server experiments, we simulate a Poisson process, with λ set to queries per second (qps).
    LAMBDA_QPS_ARRAY = config["LAMBDA_QPS_ARRAY"]

    LLM_MODELS = config["LLM_MODELS"]

################### CLASSES #####################

class EnhancedMonitorThread(threading.Thread):
    """
    Enhanced monitoring thread that tracks GPU, CPU metrics plus LLM-specific metrics
    aligned with NVIDIA NIM and vLLM standards.
    
    Works with both server (AsyncLLMEngine) and batch (LLM) workloads.
    
    Metric name changes from original MonitorThread:
    - No changes to existing metric names
    
    New LLM-specific metrics added:
    - time_to_first_token_seconds_events: List of TTFT measurements per request
    - e2e_request_latency_seconds_events: List of end-to-end latencies per request
    - request_success_total: Counter of successful requests
    - request_failure_total: Counter of failed requests
    - prompt_tokens_total: Counter of prompt tokens processed
    - generation_tokens_total: Counter of generation tokens
    - tokens_per_second_events: List of per-request throughput measurements
    - generation_tokens_total_samples: Generation token counter sampled over time
    - prompt_tokens_total_samples: Prompt token counter sampled over time
    """
    def __init__(self, gpu_indices=None, secs_between_samples=0.1, llm_engine=None):
        super().__init__(daemon=True)
        
        if gpu_indices:
            self.gpu_indices = gpu_indices
        else:
            # Monitor all GPUs - will use GPU_INDICES from utils if available
            # Otherwise default to GPU 0
            import sys
            if 'utils' in sys.modules:
                self.gpu_indices = getattr(sys.modules['utils'], 'GPU_INDICES', [0])
            else:
                self.gpu_indices = [0]
        
        self.secs_between_samples = secs_between_samples
        self.running = True
        self.gpu_handles = {i: nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices}
        self.llm_engine = llm_engine
        
        # Original metrics (unchanged names)
        self.gpu_utilization = {i: [] for i in self.gpu_indices}
        self.gpu_mem_utilization = {i: [] for i in self.gpu_indices}
        self.gpu_power_draw = {i: [] for i in self.gpu_indices}
        self.gpu_temp = {i: [] for i in self.gpu_indices}
        self.gpu_clock = {i: [] for i in self.gpu_indices}
        self.cpu_utilization = []
        self.ram_utilization = []
        
        # New LLM-specific metrics (NVIDIA NIM / vLLM aligned)
        # These are per-request events collected as they complete
        self.time_to_first_token_seconds_events = []
        self.e2e_request_latency_seconds_events = []
        self.tokens_per_second_events = []
        
        # These are counters
        self.request_success_total = 0
        self.request_failure_total = 0
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        
        # These are sampled periodically in run() loop (same frequency as GPU/CPU)
        self.generation_tokens_total_samples = []
        self.prompt_tokens_total_samples = []
        
        # Internal tracking for derived metrics
        self._lock = threading.Lock()
        self._last_snapshot_time = time.time()
        self._last_generation_tokens = 0
        
    def run(self):
        while self.running:
            for i in self.gpu_indices:
                handle = self.gpu_handles[i]
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                # Convert power to Watts
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = nvmlDeviceGetTemperature(handle, 0)
                clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
                self.gpu_utilization[i].append(util.gpu)
                self.gpu_mem_utilization[i].append(mem.used / (1024 * 1024))  # in MB
                self.gpu_power_draw[i].append(power)
                self.gpu_temp[i].append(temp)
                self.gpu_clock[i].append(clock)
            
            self.ram_utilization.append(psutil.virtual_memory().used / (1024 * 1024))
            self.cpu_utilization.append(psutil.cpu_percent())
            
            # Sample LLM metrics at same frequency as GPU/CPU
            with self._lock:
                self.generation_tokens_total_samples.append(self.generation_tokens_total)
                self.prompt_tokens_total_samples.append(self.prompt_tokens_total)
            
            time.sleep(self.secs_between_samples)
    
    def stop(self):
        self.running = False
        self.join()
    
    def update_llm_metrics(self, ttft=None, e2e_latency=None, success=False, failure=False,
                          prompt_tokens=0, generation_tokens=0, tokens_per_sec=None):
        """
        Update LLM-specific metrics. Call this from request processing.
        
        Args:
            ttft: Time to first token in seconds (per-request event)
            e2e_latency: End-to-end request latency in seconds (per-request event)
            success: Whether request succeeded
            failure: Whether request failed
            prompt_tokens: Number of prompt tokens (accumulates to counter)
            generation_tokens: Number of generation tokens (accumulates to counter)
            tokens_per_sec: Tokens per second for this request (per-request event)
        """
        with self._lock:
            # Per-request events
            if ttft is not None:
                self.time_to_first_token_seconds_events.append(ttft)
            if e2e_latency is not None:
                self.e2e_request_latency_seconds_events.append(e2e_latency)
            if tokens_per_sec is not None:
                self.tokens_per_second_events.append(tokens_per_sec)
            
            # Counters
            if success:
                self.request_success_total += 1
            if failure:
                self.request_failure_total += 1
            if prompt_tokens > 0:
                self.prompt_tokens_total += prompt_tokens
            if generation_tokens > 0:
                self.generation_tokens_total += generation_tokens
    
    def get_all_metrics(self):
        """
        Returns all metrics including original and new LLM-specific metrics.
        Original metric names are unchanged.
        """
        all_metrics = {}
        
        # Original metrics (unchanged names)
        for i in self.gpu_indices:
            all_metrics[f"gpu_{i}_memory_used_mb"] = self.gpu_mem_utilization[i]
            all_metrics[f"gpu_{i}_utilization_percent"] = self.gpu_utilization[i]
            all_metrics[f"gpu_{i}_power_draw_watts"] = self.gpu_power_draw[i]
        all_metrics["cpu_memory_used_mb"] = self.ram_utilization
        all_metrics["cpu_utilization_percent"] = self.cpu_utilization
        
        # New LLM-specific metrics
        with self._lock:
            # Per-request events (collected as they complete)
            all_metrics["time_to_first_token_seconds_events"] = self.time_to_first_token_seconds_events.copy()
            all_metrics["e2e_request_latency_seconds_events"] = self.e2e_request_latency_seconds_events.copy()
            all_metrics["tokens_per_second_events"] = self.tokens_per_second_events.copy()
            
            # Counter totals (final values)
            all_metrics["request_success_total"] = self.request_success_total
            all_metrics["request_failure_total"] = self.request_failure_total
            all_metrics["prompt_tokens_total"] = self.prompt_tokens_total
            all_metrics["generation_tokens_total"] = self.generation_tokens_total
            
            # Time-series sampled at same frequency as GPU/CPU metrics
            all_metrics["generation_tokens_total_samples"] = self.generation_tokens_total_samples.copy()
            all_metrics["prompt_tokens_total_samples"] = self.prompt_tokens_total_samples.copy()
        
        return all_metrics

################ STATIC METHODS ################

# Determinism
def fix_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wait_for_gpu_cooldown(
    gpu_handle,
    idle_time=60,          # seconds to wait at idle
    check_interval=5,       # seconds
    util_threshold=1,       # %
    power_stability_w=3,    # watts
):
    """
    Wait until the GPU has been idle and power-stable for a fixed duration.
    """

    start_temp = nvmlDeviceGetTemperature(gpu_handle, 0)
    print(f"GPU temperature before cooldown: {start_temp}°C")

    stable_start = None
    last_power = None

    while True:
        util = nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        power = nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0  # mW → W
        temp = nvmlDeviceGetTemperature(gpu_handle, 0)

        print(
            f"Cooldown check | Temp: {temp}°C | Util: {util}% | Power: {power:.1f} W"
        )

        idle = (util <= util_threshold)

        power_stable = (last_power is not None and abs(power - last_power) <= power_stability_w)

        if idle and power_stable:
            if stable_start is None:
                stable_start = time.time()
            elif time.time() - stable_start >= idle_time:
                print(
                    f"GPU idle and stable for {idle_time}s. "
                    f"Final temp: {temp}°C. Continue."
                )
                break
        else:
            stable_start = None

        last_power = power
        time.sleep(check_interval)


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

def create_vllm(model_name, async_mode=False):
    """
    Create a vLLM instance (sync or async).
    
    Args:
        model_name: Name/path of the model to load
        async_mode: If True, returns AsyncLLMEngine; if False, returns LLM
    
    Returns:
        LLM or AsyncLLMEngine instance
    """
    dtype = "auto"  # Default dtype
    device_name = "CPU"
    gpu_memory_utilization = 0.9  # Default value
    
    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Memory utilization for GPUs with <20GB VRAM is 0.8, for other devices we use 0.9.
        # Without this, most models crash on GPUs with low VRAM.
        if total_vram_gb < 20:
            gpu_memory_utilization = 0.8
            print(f"GPU VRAM: {total_vram_gb:.2f}GB. Setting gpu_memory_utilization=0.8")
        else:
            print(f"GPU VRAM: {total_vram_gb:.2f}GB. Setting gpu_memory_utilization=0.9")
        
        # Set dtype based on the detected GPU name
        if any(gpu in device_name for gpu in ["V100", "T4"]):
            print(f"Setting dtype to float16 because device does not support bfloat.")
            dtype = "float16"
    
    print(f"Detected device: {device_name}.")
    print(f"Using dtype '{dtype}' for model '{model_name}'.")
    
    # Common benchmark parameters
    llm_args = {
        "model": model_name,
        "dtype": dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": CONTEXT_LENGTH,  # input context
        # We leave scheduler policy, KV cache block, tensor parallelism, paged attention defaults
        # to vLLM defaults.
    }
    
    try:
        if async_mode:
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm import EngineArgs
            
            print("Attempting to load model in async mode with default settings...")
            engine_args = EngineArgs(**llm_args)
            llm = AsyncLLMEngine.from_engine_args(engine_args)
        else:         
            llm = LLM(**llm_args)
        
        return llm
    except RuntimeError as e:
        error_message = str(e)
        # Re-raise the exception if it's not the one we can handle
        raise e

def get_sampling_params():
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=256,
        repetition_penalty=1.0
    )