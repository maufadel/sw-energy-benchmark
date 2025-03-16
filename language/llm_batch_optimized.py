import time
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
import threading

from vllm import LLM, SamplingParams
from energymeter import EnergyMeter
from datasets import load_dataset
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# Configuration
ITERATIONS = 3

# Load dataset
ds = load_dataset("launch/open_question_type")["train"]["question"][:1000]

# Load vLLM model
llm = LLM(model="microsoft/Phi-3.5-mini-instruct", dtype="float16", 
          max_model_len=4096, gpu_memory_utilization=0.6)

# Sampling parameters
sampling_params = SamplingParams(max_tokens=500, temperature=0.7)

results = []
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# Run benchmarking
for t in range(ITERATIONS):
    print(f"Start iteration {t}")

    # Initialize energy meter
    meter = EnergyMeter(disk_avg_speed=1600 * 1e6, 
                        disk_active_power=6, 
                        disk_idle_power=1.42, 
                        label="Batch LLM", include_idle=True)

    processed_queries = 0
    total_generated_tokens = 0

    # Start monitoring
    monitor = utils.MonitorThread(handle)
    monitor.start()

    # Start energy measurement
    start_time = time.time()
    meter.begin()

    # Process all queries
    outputs = llm.generate(ds, sampling_params)

    for j, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        num_tokens = len(generated_text.split())  # Approximate token count
        total_generated_tokens += num_tokens
        
    print(f"Queries processed: {len(outputs)}, total tokens: {total_generated_tokens}\n")

    # Stop monitoring
    monitor.stop()

    # Stop energy meter
    meter.end()
    print("Simulation complete.")

    # Store results
    res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
    res["measurement_duration"] = meter.meter.result.duration / 1000000
    res["measurement_timestamp"] = meter.meter.result.timestamp
    res["measurement_datetime"] = datetime.fromtimestamp(meter.meter.result.timestamp,
                                                          datetime.now().astimezone().tzinfo).isoformat()
    res["sampling_params"] = sampling_params.__dict__
    res["model"] = "microsoft/Phi-3.5-mini-instruct"
    res["processed_queries"] = len(ds)
    res["total_generated_tokens"] = total_generated_tokens
    res.update(monitor.get_all_metrics())
    results.append(res)

    # Save results
    pd.DataFrame(results).to_csv("results/llm_batch_optimized_results.csv")
    utils.wait_for_gpu_cooldown(handle)

nvmlShutdown()
