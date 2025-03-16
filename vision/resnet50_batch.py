import time
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
import threading

from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from energymeter import EnergyMeter
from datasets import load_dataset
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# Configuration
ITERATIONS = utils.ITERATIONS

# Load dataset
dataset = load_dataset("Kaludi/data-food-classification", trust_remote_code=True)
images = dataset["train"]["image"][:1000]

# Load processor and model, and move model to GPU if available
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
if torch.cuda.device_count() > 0:   
    model.to("cuda")

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
                        label="Batch Resnet50", include_idle=True)

    predicted_labels = []

    # Start monitoring
    monitor = utils.MonitorThread(handle, secs_between_samples=0.1)
    monitor.start()

    # Start energy measurement
    start_time = time.time()
    meter.begin()

    for i in range(10000):
        # Process image and move inputs to GPU
        image = images[i % len(images)]
        inputs = processor(image, return_tensors="pt")
        if torch.cuda.device_count() > 0:   
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
        with torch.no_grad():
            logits = model(**inputs).logits
        if torch.cuda.device_count() > 0:   
            torch.cuda.synchronize()  # Ensure GPU ops are finished
    
        predicted_labels.append(logits.argmax(-1).item())
        
    print(f"Queries processed: {len(predicted_labels)}\n")

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
    res["model"] = "microsoft/resnet-50"
    res["processed_queries"] = len(predicted_labels)
    res.update(monitor.get_all_metrics())
    results.append(res)

    # Save results
    pd.DataFrame(results).to_csv("results/resnet50_batch_results.csv")
    utils.wait_for_gpu_cooldown(handle)

nvmlShutdown()
