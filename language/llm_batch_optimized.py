import time
import gc
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown
import threading
import traceback

from vllm import LLM, SamplingParams
from energymeter import EnergyMeter
from datasets import load_dataset
import sys
import os
import argparse
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# Configuration
ITERATIONS = utils.ITERATIONS
MODELS = utils.LLM_MODELS

# Load dataset
ds = load_dataset("launch/open_question_type")["train"]["question"][:1000]

# Sampling parameters
sampling_params = SamplingParams(max_tokens=500, temperature=0.7)

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

all_results = []
query_log = []

parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, nargs='?', default="results",
                help="Path to the result folder (default: 'results')")
args = parser.parse_args()
result_folder_path = args.result_folder
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)
    print(f"Created directory: {result_folder_path}")

print(f"The results will be saved in: {result_folder_path}")

# Run benchmarking for each model
for model_name in MODELS:
    llm_loaded = False
    try:
        print(f"Loading model {model_name}")
        llm = utils.create_vllm(model_name)
        llm_loaded = True
        
        for t in range(ITERATIONS):
            print(f"Start iteration {t} for model {model_name}")
    
            # Initialize energy meter
            meter = EnergyMeter(label="Batch LLM", include_idle=True, ignore_disk=True)
    
            processed_queries = 0
            total_generated_tokens = 0
    
            # Start monitoring
            monitor = utils.MonitorThread()
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
    
                query_log.append({
                    "iteration": t,
                    "model": model_name,
                    "query_id": j,
                    "response_text": generated_text,
                    "num_tokens": num_tokens,
                    "timestamp": datetime.now().isoformat()
                })
    
            print(f"Queries processed: {len(outputs)}, total tokens: {total_generated_tokens}\n")
    
            # Stop monitoring
            monitor.stop()
    
            # Stop energy meter
            meter.end()
            print("Simulation complete.")
    
            # Store results
            res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
            res["measurement_duration"] = meter.duration
            res["measurement_timestamp"] = meter.start_time
            res["measurement_datetime"] = datetime.fromtimestamp(res["measurement_timestamp"], 
                                                                 datetime.now().astimezone().tzinfo).isoformat()
            res["sampling_params"] = sampling_params.__dict__
            res["model"] = model_name
            res["processed_queries"] = len(ds)
            res["total_generated_tokens"] = total_generated_tokens
            res.update(monitor.get_all_metrics())
            all_results.append(res)
    
            # Save results
            pd.DataFrame(all_results).to_csv(result_folder_path+"/llm_batch_optimized_results.csv", index=False)
            pd.DataFrame(query_log).to_csv(result_folder_path+"/llm_batch_optimized_details.csv", index=False)
            
            utils.wait_for_gpu_cooldown(handle)

    except Exception as e:
        print(f"Error with model {model_name}")
        print(traceback.format_exc())
    finally:
        # Delete the llm object and free the memory
        if llm_loaded:
            del llm
        torch.cuda.empty_cache()
        gc.collect()

nvmlShutdown()
