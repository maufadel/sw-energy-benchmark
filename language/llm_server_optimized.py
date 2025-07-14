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
import argparse
from copy import deepcopy
import traceback

import signal

def signal_handler(signum, frame):
    """
    This function is called when the script receives a signal.
    It prints the current stack trace and then exits.
    """
    print(f"Received signal: {signum}", file=sys.stderr)
    print("Printing stack traceback...", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)
    sys.exit(1)

# Register the signal handler for SIGUSR1
signal.signal(signal.SIGUSR1, signal_handler)



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

            if len(queries) > 0:
                try:
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
                except Exception as e:
                    print("Error when doing inference", e)
                    print(traceback(e))

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

    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder", type=str, nargs='?', default="results",
                    help="Path to the result folder (default: 'results')")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to the config file (default: 'config.yaml')")
    args = parser.parse_args()
    result_folder_path = args.result_folder
    utils.load_config(args.config)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
        print(f"Created directory: {result_folder_path}")
    
    print(f"The results will be saved in: {result_folder_path}")

    for model_name in MODELS:
        llm_loaded = False
        try:
            print(f"Loading model {model_name}")
            llm = utils.create_vllm(model_name)
            llm_loaded = True
            for lambda_qps in LAMBDA_QPS_ARRAY:
                for t in range(ITERATIONS):
                    print(f"Start iteration {t} with model {model_name} and lambda_qps {lambda_qps}")
                    monitor = utils.MonitorThread()
                    query_queue = queue.Queue()
                    result_lock = threading.Lock()
    
                    inference_thread = InferenceThread(llm, dataset, sampling_params, query_queue, result_lock, query_log)
                    query_generator = QueryGeneratorThread(query_queue, lambda_qps, model_name)
                    meter = EnergyMeter(label="Chatbot", include_idle=True, ignore_disk=True)
    
                    monitor.start()
                    inference_thread.start()
                    query_generator.start()
                    meter.begin()
    
                    time.sleep(TEST_DURATION)

                    # Stop threads and wait for them to finish.
                    query_generator.stop()
                    inference_thread.stop()
                    query_generator.join()
                    inference_thread.join()
                    
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
                    pd.DataFrame(results).to_csv(result_folder_path+"/llm_server_optimized_results.csv")
                    pd.DataFrame(query_log).to_csv(result_folder_path+"/llm_server_optimized_details.csv")
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
