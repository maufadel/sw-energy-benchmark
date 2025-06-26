import threading
import gc
import torch
import time
import queue
import numpy as np
import psutil
import pandas as pd
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown
from vllm import LLM, SamplingParams
from energymeter import EnergyMeter
from datasets import load_dataset
import sys
import os
from copy import deepcopy
import traceback

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# CONSTANTS
ITERATIONS = utils.ITERATIONS
TEST_DURATION = utils.MAX_TEST_DURATION
LAMBDA_QPS_ARRAY = utils.LAMBDA_QPS_ARRAY
MODELS = utils.LLM_MODELS

class InferenceThread(threading.Thread):
    def __init__(self, llm, dataset, sampling_params, query_queue, result_lock, query_log):
        super().__init__(daemon=True)
        self.llm = llm
        self.dataset = dataset
        self.sampling_params = sampling_params
        self.query_queue = query_queue
        self.result_lock = result_lock
        self.query_log = query_log
        self.total_generated_tokens = 0
        self.processed_queries = 0
        self.running = True

    def run(self):
        while self.running:
            queries = []
            metadata = []
            while not self.query_queue.empty():
                item = self.query_queue.get()
                if item is not None:
                    queries.append(item["query"])
                    metadata.append(item)
                    self.query_queue.task_done()

            if queries:
                start_time = datetime.now()
                outputs = self.llm.generate([self.dataset[q % len(self.dataset)] for q in queries], 
                                            self.sampling_params)
                end_time = datetime.now()

                with self.result_lock:
                    for output, meta in zip(outputs, metadata):
                        response_text = output.outputs[0].text
                        self.total_generated_tokens += len(response_text.split())
                        self.processed_queries += 1

                        self.query_log.append({
                            "query_id": meta["query"],
                            "model": meta["model"],
                            "lambda_qps": meta["lambda_qps"],
                            "queued_time": meta["queued_time"].isoformat(),
                            "inference_start": start_time.isoformat(),
                            "inference_end": end_time.isoformat(),
                            "response_text": response_text,
                        })

    def stop(self):
        self.running = False

class QueryGeneratorThread(threading.Thread):
    def __init__(self, query_queue, lambda_qps, model_name):
        super().__init__(daemon=True)
        self.query_queue = query_queue
        self.lambda_qps = lambda_qps
        self.model_name = model_name
        self.queries_generated = 0
        self.running = True

    def run(self):
        query_id = 0
        while self.running:
            time.sleep(np.random.exponential(1 / self.lambda_qps))
            query_id += 1
            self.queries_generated += 1
            self.query_queue.put({
                "query": query_id,
                "model": self.model_name,
                "lambda_qps": self.lambda_qps,
                "queued_time": datetime.now()
            })

    def stop(self):
        self.running = False

if __name__ == "__main__":
    results = []
    query_log = []
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    sampling_params = SamplingParams(max_tokens=500, temperature=0.7)
    dataset = load_dataset("launch/open_question_type")['train']['question']

    for model_name in MODELS:
        llm_loaded = False
        try:
            print(f"Loading model {model_name}")
            llm = LLM(model=model_name, dtype="auto")
            llm_loaded = True
            for lambda_qps in LAMBDA_QPS_ARRAY:
                for t in range(ITERATIONS):
                    print(f"Start iteration {t} with model {model_name} and lambda_qps {lambda_qps}")
                    monitor = utils.MonitorThread()
                    query_queue = queue.Queue()
                    result_lock = threading.Lock()
    
                    inference_thread = InferenceThread(llm, dataset, sampling_params, query_queue, result_lock, query_log)
                    query_generator = QueryGeneratorThread(query_queue, lambda_qps, model_name)
                    meter = EnergyMeter(disk_avg_speed=1600 * 1e6, disk_active_power=6, 
                                        disk_idle_power=1.42, label="Chatbot", include_idle=True)
    
                    monitor.start()
                    inference_thread.start()
                    query_generator.start()
                    meter.begin()
    
                    time.sleep(TEST_DURATION)
    
                    query_generator.stop()
                    inference_thread.stop()
                    monitor.stop()
                    meter.end()
    
                    print(f"Processed {inference_thread.processed_queries} queries")
                    res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
                    res["measurement_duration"] = meter.duration
                    res["measurement_timestamp"] = meter.start_time
                    res["measurement_datetime"] = datetime.fromtimestamp(res["measurement_timestamp"], 
                                                                         datetime.now().astimezone().tzinfo).isoformat()
                    res["sampling_params"] = sampling_params.__dict__
                    res["model"] = model_name
                    res["lambda_qps"] = lambda_qps
                    res.update(monitor.get_all_metrics())
                    res.update({
                        "total_generated_tokens": inference_thread.total_generated_tokens,
                        "processed_queries": inference_thread.processed_queries,
                        "queries_generated": query_generator.queries_generated,
                    })
                    results.append(res)
                    pd.DataFrame(results).to_csv("results/llm_server_optimized_results.csv")
                    pd.DataFrame(query_log).to_csv("results/llm_server_optimized_details.csv")
                    del inference_thread
                    del query_generator
    
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
