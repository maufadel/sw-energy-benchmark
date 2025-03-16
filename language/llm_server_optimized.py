import threading
import time
import queue
import numpy as np
import psutil
import pandas as pd
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
from vllm import LLM, SamplingParams
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
    def __init__(self, llm, dataset, sampling_params, query_queue, result_lock):
        super().__init__(daemon=True)
        self.llm = llm
        self.dataset = dataset
        self.sampling_params = sampling_params
        self.query_queue = query_queue
        self.result_lock = result_lock
        self.total_generated_tokens = 0
        self.processed_queries = 0
        self.running = True
    
    def run(self):
        while self.running:
            queries = []
            while not self.query_queue.empty():
                queries.append(self.query_queue.get())
                self.query_queue.task_done()
            
            if queries:
                outputs = self.llm.generate([self.dataset[q % len(self.dataset)] for q in queries], 
                                            self.sampling_params)
                with self.result_lock:
                    for output in outputs:
                        self.total_generated_tokens += len(output.outputs[0].text.split())
                        self.processed_queries += 1

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
    
    def stop(self):
        self.running = False

if __name__ == "__main__":
    results = []
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    llm = LLM(model="microsoft/Phi-3.5-mini-instruct", dtype="float16", max_model_len=4096, gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(max_tokens=500, temperature=0.7)
    dataset = load_dataset("launch/open_question_type")['train']['question']
    query_queue = queue.Queue()
    result_lock = threading.Lock()

    for lambda_qps in LAMBDA_QPS_ARRAY:
        for t in range(ITERATIONS):
            print(f"Start iteration {t} with lambda_qps {lambda_qps}")
            monitor = utils.MonitorThread(handle)
            inference_thread = InferenceThread(llm, dataset, sampling_params, query_queue, result_lock)
            query_generator = QueryGeneratorThread(query_queue, lambda_qps)
            meter = EnergyMeter(disk_avg_speed=1600 * 1e6, disk_active_power=6, 
                                disk_idle_power=1.42, label="Chatbot", include_idle=True)
    
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
            res["sampling_params"] = sampling_params.__dict__
            res["model"] = "microsoft/Phi-3.5-mini-instruct"
            res["lambda_qps"] = lambda_qps
            res.update(monitor.get_all_metrics())
            res.update({
                "total_generated_tokens": inference_thread.total_generated_tokens,
                "processed_queries": inference_thread.processed_queries,
                "queries_generated": query_generator.queries_generated,
            })
            results.append(res)
            pd.DataFrame(results).to_csv("results/llm_server_optimized_results.csv")

            # Wait for cooldown.
            utils.wait_for_gpu_cooldown(handle)

    nvmlShutdown()
