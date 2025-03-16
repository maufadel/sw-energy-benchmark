import threading
import time
import queue
import numpy as np
import psutil
import pandas as pd
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from energymeter import EnergyMeter
from datasets import load_dataset
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# CONSTANTS
ITERATIONS = 3
TEST_DURATION = 5*60 #secs
# We simulate a Poisson process, with λ set to queries per second (qps).
# Average monthly views per website 375773, top 0.5% websites have more than 10M monthly views.
# Source: https://blog.hubspot.com/website/web-traffic-analytics-report
# qps = monthly views / days / hours / min / secs
LAMBDA_QPS_ARRAY = [375773 / 30 / 24 / 60 / 60, 10000000 / 30 / 24 / 60 / 60]

class InferenceThread(threading.Thread):
    def __init__(self, model, processor, dataset, query_queue, result_lock):
        super().__init__(daemon=True)
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.query_queue = query_queue
        self.result_lock = result_lock
        self.processed_queries = 0
        self.running = True
    
    def run(self):
        while self.running:
            queries = []
            while not self.query_queue.empty():
                queries.append(self.query_queue.get())
                self.query_queue.task_done()
            
            if queries:
                outputs = []
                for q in queries:
                    image = self.dataset[q % len(self.dataset)]
                    inputs = self.processor(image, return_tensors="pt")
                    if torch.cuda.device_count() > 0:   
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                    if torch.cuda.device_count() > 0:   
                        torch.cuda.synchronize()  # Ensure GPU ops are finished
                
                    outputs.append(logits.argmax(-1).item())
                    
                with self.result_lock:
                    self.processed_queries += len(outputs)

    def stop(self):
        self.running = False

class QueryGeneratorThread(threading.Thread):
    def __init__(self, query_queue, lambda_qps):
        super().__init__(daemon=True)
        self.query_queue = query_queue
        self.lambda_qps = lambda_qps
        self.queries_generated = 0
        self.running = True
    
    def run(self):
        query_id = 0
        while self.running:
            time.sleep(np.random.exponential(1 / self.lambda_qps))
            query_id += 1
            self.queries_generated += 1
            self.query_queue.put(query_id)
            print(f"New query arrived, query length: {self.query_queue.qsize()}")
    
    def stop(self):
        self.running = False

if __name__ == "__main__":
    results = []
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    # Load dataset
    dataset = load_dataset("Kaludi/data-food-classification", trust_remote_code=True)["train"]["image"][:1000]
    
    # Load processor and model, and move model to GPU if available
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    if torch.cuda.device_count() > 0:   
        model.to("cuda")
    query_queue = queue.Queue()
    result_lock = threading.Lock()

    for lambda_qps in LAMBDA_QPS_ARRAY:
        for t in range(ITERATIONS):
            print(f"Start iteration {t} with lambda_qps {lambda_qps}")
            monitor = utils.MonitorThread(handle)
            inference_thread = InferenceThread(model, processor, dataset, query_queue, result_lock)
            query_generator = QueryGeneratorThread(query_queue, lambda_qps)
            meter = EnergyMeter(disk_avg_speed=1600 * 1e6, disk_active_power=6, 
                                disk_idle_power=1.42, label="Resnet50 Server", include_idle=True)
    
            monitor.start()
            inference_thread.start()
            query_generator.start()
            meter.begin()
            
            time.sleep(TEST_DURATION)  # Run for test duration
    
            query_generator.stop()
            inference_thread.stop()
            monitor.stop()
            meter.end()
    
            print(f"Processed {inference_thread.processed_queries} queries")
            res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
            res["measurement_duration"] = meter.meter.result.duration / 1000000
            res["measurement_timestamp"] = meter.meter.result.timestamp
            res["measurement_datetime"] = datetime.fromtimestamp(meter.meter.result.timestamp,
                                                                  datetime.now().astimezone().tzinfo).isoformat()
            res["model"] = "microsoft/resnet-50"
            res["lambda_qps"] = lambda_qps
            res.update(monitor.get_all_metrics())
            res.update({
                "processed_queries": inference_thread.processed_queries,
                "queries_generated": query_generator.queries_generated,
            })
            results.append(res)
            pd.DataFrame(results).to_csv("results/resnet50_server_results.csv")

            # Wait for cooldown.
            utils.wait_for_gpu_cooldown(handle)

    nvmlShutdown()
