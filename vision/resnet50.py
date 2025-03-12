from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from energymeter import EnergyMeter
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Load dataset
dataset = load_dataset("Kaludi/data-food-classification", trust_remote_code=True)
images = dataset["train"]["image"]

# Load processor and model, and move model to GPU if available
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
if torch.cuda.is_available():
    model.to("cuda")

results = []

for i in range(50):
    meter = EnergyMeter(disk_avg_speed=1600*1e6, 
                        disk_active_power=6, 
                        disk_idle_power=1.42, 
                        # inference on GPU is too fast, so we might miss the GPU utilization because
                        # of sampling and no data will be used, so we set idle=True.
                        label="Resnet50", include_idle=True) 

    meter.begin()
    
    # Process image and move inputs to GPU
    image = images[i]
    inputs = processor(image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure GPU ops are finished

    predicted_label = logits.argmax(-1).item()
    meter.end()

    res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
    res["measurement_duration"] = meter.meter.result.duration / 1000000
    res["measurement_timestamp"] = meter.meter.result.timestamp
    res["measurement_datetime"] = datetime.fromtimestamp(
        meter.meter.result.timestamp, datetime.now().astimezone().tzinfo
    ).isoformat()
    res["predicted_class"] = predicted_label
    res["index_image"] = i
    results.append(res)
    
    time.sleep(10)

pd.DataFrame(results).to_csv("results/resnet50_benchmark_results.csv")