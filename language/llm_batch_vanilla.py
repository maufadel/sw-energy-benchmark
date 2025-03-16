import time
from datetime import datetime

from energymeter import EnergyMeter
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# Configuration
ITERATIONS = 3
MAX_TEST_DURATION = 5*60

# Load dataset
ds = load_dataset("launch/open_question_type")["train"]["question"][:1000]

# Load vLLM model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "do_sample": False,
}

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
    start_time = time.time()
    meter.begin()

    # Process queries
    query_start_time = time.time()

    for i, q in enumerate(ds):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": ds[i]}
        ]
        output = pipe(messages, **generation_args)
    
        generated_text = output[0]['generated_text']
        num_tokens = len(generated_text.split())  # Approximate token count
        total_generated_tokens += num_tokens

        print(f"Queries processed: {i}, total tokens: {total_generated_tokens}\n")

        if time.time() - start_time >= MAX_TEST_DURATION:
            print(f"Maximum test duration of {MAX_TEST_DURATION} seconds reached. Stopping.")
            break

    # Stop monitoring
    monitor.stop()
    meter.end()
    print("Simulation complete.")

    # Store results
    res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
    res["measurement_duration"] = meter.meter.result.duration / 1000000
    res["measurement_timestamp"] = meter.meter.result.timestamp
    res["measurement_datetime"] = datetime.fromtimestamp(meter.meter.result.timestamp,
                                                          datetime.now().astimezone().tzinfo).isoformat()
    res["sampling_params"] = generation_args
    res["model"] = "microsoft/Phi-3.5-mini-instruct"
    res["processed_queries"] = len(ds)
    res["total_generated_tokens"] = total_generated_tokens
    res.update(monitor.get_all_metrics())
    results.append(res)

    # Save results
    pd.DataFrame(results).to_csv("results/llm_batch_vanilla_results.csv")
    utils.wait_for_gpu_cooldown(handle)  # Cooldown before next iteration

nvmlShutdown()
