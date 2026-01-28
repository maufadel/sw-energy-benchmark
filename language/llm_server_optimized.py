import gc
import torch
import time
import numpy as np
import psutil
import pandas as pd
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from energymeter import EnergyMeter
from datasets import load_dataset
import sys
import os
import argparse
import traceback
import asyncio
import threading
from collections import defaultdict
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

import signal


def signal_handler(signum, frame):
    """
    This function is called when the script receives a signal.
    It prints the current stack trace and attempts cleanup before exit.
    """
    signal_names = {
        signal.SIGTERM: "SIGTERM",
        signal.SIGINT: "SIGINT",
        signal.SIGUSR1: "SIGUSR1"
    }
    sig_name = signal_names.get(signum, f"Signal {signum}")

    print(f"\nReceived {sig_name} - Initiating graceful shutdown", file=sys.stderr)
    print("Stack traceback:", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)

    # Attempt cleanup if llm exists in globals
    if 'llm' in globals():
        print("\nAttempting cleanup of active LLM...", file=sys.stderr)
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import utils
        utils.cleanup_vllm_resources(
            llm=globals()['llm'],
            model_name=globals().get('current_model_name', 'unknown'),
            verbose=True
        )

    print(f"\nExiting due to {sig_name}", file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils

# CONSTANTS
ITERATIONS = None
TEST_DURATION = None
LAMBDA_QPS_ARRAY = None
LLM_MODELS = None
WARMUP_DURATION = None


async def process_single_request(llm, prompt, sampling_params, request_id, query_id, queued_time, 
                                model_name, lambda_qps, monitor, iteration):
    """Process a single request using AsyncLLMEngine.generate() correctly."""    
    try:        
        # engine.generate() returns an async generator - must iterate over it
        results_generator = llm.generate(prompt, sampling_params, request_id)
        
        final_output = None
        
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None:
            raise RuntimeError(f"No output received for request {request_id}")
        
        response_text = final_output.outputs[0].text
        num_generation_tokens = len(final_output.outputs[0].token_ids)
        prompt_tokens = len(final_output.prompt_token_ids)
        
        # Extract metrics from vLLM's RequestStateStats
        request_metrics = final_output.metrics
        
        # Use vLLM's built-in metrics
        ttft = request_metrics.first_token_latency if hasattr(request_metrics, 'first_token_latency') else None
        
        # Calculate e2e latency from vLLM timestamps
        # See more here https://docs.vllm.ai/en/stable/design/metrics/#engine-core-events
        if hasattr(request_metrics, 'queued_ts') and hasattr(request_metrics, 'last_token_ts'):
            # Convert monotonic time to relative duration
            e2e_latency = request_metrics.last_token_ts - request_metrics.queued_ts
        
        tokens_per_sec = num_generation_tokens / e2e_latency if e2e_latency > 0 else 0
        
        # Calculate queue time (time spent waiting before scheduling)
        queue_time = None
        if hasattr(request_metrics, 'scheduled_ts') and hasattr(request_metrics, 'queued_ts'):
            queue_time = request_metrics.scheduled_ts - request_metrics.queued_ts
        
        # Update LLM-specific metrics
        monitor.update_llm_metrics(
            ttft=ttft,
            e2e_latency=e2e_latency,
            success=True,
            prompt_tokens=prompt_tokens,
            generation_tokens=num_generation_tokens,
            tokens_per_sec=tokens_per_sec
        )
        
        return {
            "iteration": iteration,
            "query_id": query_id,
            "model": model_name,
            "lambda_qps": lambda_qps,
            "queued_time": queued_time.isoformat(),
            "response_text": response_text,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": num_generation_tokens,
            "ttft_seconds": ttft,
            "e2e_latency_seconds": e2e_latency,
            "queue_time_seconds": queue_time,
            "tokens_per_second": tokens_per_sec,
            "arrival_time": request_metrics.arrival_time if hasattr(request_metrics, 'arrival_time') else None,
            "queued_ts": request_metrics.queued_ts if hasattr(request_metrics, 'queued_ts') else None,
            "scheduled_ts": request_metrics.scheduled_ts if hasattr(request_metrics, 'scheduled_ts') else None,
            "first_token_ts": request_metrics.first_token_ts if hasattr(request_metrics, 'first_token_ts') else None,
            "last_token_ts": request_metrics.last_token_ts if hasattr(request_metrics, 'last_token_ts') else None,
            "is_corrupted": request_metrics.is_corrupted if hasattr(request_metrics, 'is_corrupted') else False
        }
    except Exception as e:
        # Update failure metric
        if monitor:
            monitor.update_llm_metrics(failure=True)
        raise e


async def run_benchmark(llm, dataset, sampling_params, lambda_qps, model_name, test_duration, monitor, iteration):
    """Run async benchmark with open-loop query generation."""
    query_log = []
    total_generated_tokens = 0
    processed_queries = 0
    queries_submitted = 0
    
    start_test = time.time()
    query_id = -1 # We start with index 0
    
    # List to track all pending tasks
    pending_tasks = []
    
    while time.time() - start_test < test_duration:
        # Generate next query based on Poisson process
        await asyncio.sleep(np.random.exponential(1 / lambda_qps))
        
        query_id += 1
        queries_submitted += 1
        queued_time = datetime.now()
        
        # Submit request immediately (open-loop)
        prompt = dataset[query_id % len(dataset)]
        request_id = f"request-{query_id}"
        
        # Create task for this request
        task = asyncio.create_task(
            process_single_request(
                llm, prompt, sampling_params, request_id, 
                query_id, queued_time, model_name, lambda_qps,
                monitor, iteration
            )
        )
        pending_tasks.append(task)
    
    # Wait for all pending requests to complete
    print(f"Waiting for {len(pending_tasks)} pending requests to complete...")
    results = await asyncio.gather(*pending_tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            print(f"Error in request: {result}")
        else:
            query_log.append(result)
            total_generated_tokens += result["generation_tokens"]
            processed_queries += 1
    
    return {
        "query_log": query_log,
        "total_generated_tokens": total_generated_tokens,
        "processed_queries": processed_queries,
        "queries_submitted": queries_submitted
    }


async def run_iteration(llm, dataset, sampling_params, lambda_qps, model_name, test_duration, handle, iteration):
    """Run a single iteration of the benchmark."""
    print(f"Running with model {model_name} and lambda_qps {lambda_qps}")

    monitor = utils.EnhancedMonitorThread(llm_engine=llm)
    meter = EnergyMeter(label="Chatbot", include_idle=True, ignore_disk=True)
    
    monitor.start()
    meter.begin()
    
    # Run the benchmark
    result = await run_benchmark(llm, dataset, sampling_params, lambda_qps, model_name, test_duration, monitor, iteration)

    meter.end()
    monitor.stop()
    
    print(f"Processed {result['processed_queries']} queries")
    
    # Compile results
    res = {k: np.sum(v) if isinstance(v, list) else v for k, v in meter.get_total_joules_per_component().items()}
    res["measurement_duration"] = meter.duration
    res["measurement_timestamp"] = meter.start_time
    res["measurement_datetime"] = datetime.fromtimestamp(res["measurement_timestamp"], 
                                                         datetime.now().astimezone().tzinfo).isoformat()
    res["sampling_params"] = str(sampling_params)
    res["model"] = model_name
    res["lambda_qps"] = lambda_qps
    res.update(monitor.get_all_metrics())
    res.update({
        "total_generated_tokens": result["total_generated_tokens"],
        "processed_queries": result["processed_queries"],
        "queries_submitted": result["queries_submitted"],
    })
    
    return res, result["query_log"]


async def main_async(result_folder_path, handle, sampling_params, dataset):
    """Main async function to run all benchmarks."""
    results = []
    query_log = []
    
    current_model_name = None
    failed_models = []

    for model_name in LLM_MODELS:
        current_model_name = model_name
        llm_loaded = False
        llm = None

        # Clean GPU memory before loading model to ensure clean state
        utils.cleanup_gpu_memory(verbose=True)

        try:
            print(f"Loading model {model_name}")
            engine_args = AsyncEngineArgs(model=model_name)
            llm = AsyncLLMEngine.from_engine_args(engine_args)
            llm_loaded = True

            # Warmup iteration.
            print("Warming up")
            await run_iteration(
                llm, dataset, sampling_params, max(LAMBDA_QPS_ARRAY), 
                model_name, WARMUP_DURATION, handle, -1
            )
            utils.wait_for_gpu_cooldown(handle)
            
            for lambda_qps in LAMBDA_QPS_ARRAY:
                for t in range(ITERATIONS):
                    print(f"Start iteration {t} with model {model_name} and lambda_qps {lambda_qps}")
                    
                    # Run iteration
                    res, iter_query_log = await run_iteration(
                        llm, dataset, sampling_params, lambda_qps, 
                        model_name, TEST_DURATION, handle, t
                    )
                    
                    results.append(res)
                    query_log.extend(iter_query_log)
                    
                    pd.DataFrame(results).to_csv(result_folder_path+"/llm_server_optimized_results.csv")
                    pd.DataFrame(query_log).to_csv(result_folder_path+"/llm_server_optimized_details.csv")
    
                    utils.wait_for_gpu_cooldown(handle)
    
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR: Exception occurred with model {model_name}")
            print(f"{'='*60}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
            print(f"{'='*60}\n")

            failed_models.append({
                'model': model_name,
                'error': str(e),
                'error_type': type(e).__name__
            })

        finally:
            print(f"\nExecuting cleanup for model: {model_name}")
            cleanup_success = utils.cleanup_vllm_resources(
                llm=llm if llm is not None else None,
                model_name=model_name,
                verbose=True
            )

            if not cleanup_success:
                print(f"WARNING: Cleanup completed with errors for {model_name}")
                print("This may affect subsequent models. Continuing anyway...")

            time.sleep(2)

    # Print summary of failures
    if failed_models:
        print("\n" + "="*60)
        print("SUMMARY: The following models encountered errors:")
        print("="*60)
        for failure in failed_models:
            print(f"  - {failure['model']}")
            print(f"    Error: {failure['error_type']}: {failure['error'][:100]}")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("SUCCESS: All models processed without errors")
        print("="*60 + "\n")

    return failed_models


if __name__ == "__main__":
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
    ITERATIONS = utils.ITERATIONS
    TEST_DURATION = utils.MAX_TEST_DURATION
    LAMBDA_QPS_ARRAY = utils.LAMBDA_QPS_ARRAY
    LLM_MODELS = utils.LLM_MODELS
    WARMUP_DURATION = utils.WARMUP_DURATION
    print(LLM_MODELS)

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    sampling_params = utils.get_sampling_params()
    dataset = load_dataset("rajpurkar/squad")['train']['question']

    # Run the async main function
    failed_models = asyncio.run(main_async(result_folder_path, handle, sampling_params, dataset))

    nvmlShutdown()

    if failed_models:
        print("Forcing process exit due to model failures to prevent hanging...")
        print("Exiting with error code 1...")
        os._exit(1)