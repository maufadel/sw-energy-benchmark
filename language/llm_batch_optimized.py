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
import signal
import sys
import os
import argparse
import random

from vllm import LLM, SamplingParams
from energymeter import EnergyMeter
from datasets import load_dataset
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils


def signal_handler(signum, frame):
    """
    Handle termination signals gracefully.
    Prints diagnostics and exits cleanly.
    """
    signal_names = {
        signal.SIGTERM: "SIGTERM",
        signal.SIGINT: "SIGINT",
        signal.SIGUSR1: "SIGUSR1"
    }
    sig_name = signal_names.get(signum, f"Signal {signum}")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Received {sig_name} - Initiating graceful shutdown", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print("Stack trace at signal:", file=sys.stderr)
    traceback.print_stack(frame, file=sys.stderr)

    # Attempt cleanup if llm exists in globals
    if 'llm' in globals():
        print("\nAttempting cleanup of active LLM...", file=sys.stderr)
        utils.cleanup_vllm_resources(
            llm=globals()['llm'],
            model_name=globals().get('current_model_name', 'unknown'),
            verbose=True
        )

    print(f"\nExiting due to {sig_name}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder", type=str, nargs='?', default="results",
                        help="Path to the result folder (default: 'results')")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to the config file (default: 'config.yaml')")
    args = parser.parse_args()
    result_folder_path = args.result_folder
    utils.load_config(args.config)
    utils.fix_seeds()
    # Configuration
    ITERATIONS = utils.ITERATIONS
    LLM_MODELS = utils.LLM_MODELS
    print(LLM_MODELS)
    # Load dataset
    dataset = load_dataset("rajpurkar/squad")["train"]["question"][:1000]

    # Deterministic decoding (benchmark contract)
    sampling_params = utils.get_sampling_params()

    # Register signal handlers
    # Note: SIGINT (Ctrl+C) is NOT registered here to allow natural cancellation
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    all_results = []
    query_log = []
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
        print(f"Created directory: {result_folder_path}")

    print(f"The results will be saved in: {result_folder_path}")

    # Track current model for signal handler and failed models for summary
    current_model_name = None
    failed_models = []

    # Run benchmarking for each model
    for model_name in LLM_MODELS:
        current_model_name = model_name  # Track for signal handler
        llm_loaded = False
        llm = None  # Explicitly initialize
        tokenizer = None

        # Clean GPU memory before loading model to ensure clean state
        utils.cleanup_gpu_memory(verbose=True)

        try:
            print(f"Loading model {model_name}")
            llm = utils.create_vllm(model_name)
            
            # Get tokenizer for token counting
            tokenizer = llm.get_tokenizer()
            llm_loaded = True
            
            for t in range(ITERATIONS):
                print(f"Start iteration {t} for model {model_name}")

                ############################
                # Batch warm-up (discarded)
                ############################
                warmup_prompt = dataset[:1]
                _ = llm.generate(warmup_prompt, sampling_params)

                torch.cuda.synchronize()
                time.sleep(2)

                ############################
                # Monitoring + energy
                ############################
        
                # Initialize energy meter
                meter = EnergyMeter(label="Batch LLM", include_idle=True, ignore_disk=True)
        
                # Start monitoring with EnhancedMonitorThread
                monitor = utils.EnhancedMonitorThread(llm_engine=llm)
                monitor.start()
        
                # Start energy measurement
                start_time = time.time()
                meter.begin()

                # Process all queries
                batch_start = datetime.now()
                outputs = llm.generate(dataset, sampling_params)
                batch_end = datetime.now()
                
                torch.cuda.synchronize()

                # Stop monitoring
                monitor.stop()
        
                # Stop energy meter
                meter.end()

                batch_duration = (batch_end - batch_start).total_seconds()
                
                for j, (prompt, output) in enumerate(zip(dataset, outputs)):
                    generated_text = output.outputs[0].text
                    
                    # Count tokens using tokenizer
                    in_tokens = len(tokenizer.encode(prompt))
                    out_tokens = len(tokenizer.encode(generated_text))
                    
                    # Update monitor with per-request metrics
                    # For batch processing, TTFT and e2e are approximations
                    if monitor:
                        monitor.update_llm_metrics(
                            success=True,
                            prompt_tokens=in_tokens,
                            generation_tokens=out_tokens
                        )
        
                    query_log.append({
                        "iteration": t,
                        "model": model_name,
                        "query_id": j,
                        "response_text": generated_text,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Calculate metrics
                avg_tokens_per_sec = out_tokens / batch_duration if batch_duration > 0 else 0
                if monitor:
                    monitor.update_llm_metrics(
                        e2e_latency=batch_duration,
                        tokens_per_sec=avg_tokens_per_sec
                    )
        
                print(f"Queries processed: {len(outputs)}, total output tokens: {out_tokens}\n")
        
                # Store results
                res = {k: np.sum(v) for k, v in meter.get_total_joules_per_component().items()}
                res["iteration"] = t
                res["measurement_duration"] = meter.duration
                res["measurement_timestamp"] = meter.start_time
                res["measurement_datetime"] = datetime.fromtimestamp(res["measurement_timestamp"], 
                                                                     datetime.now().astimezone().tzinfo).isoformat()
                res["sampling_params"] = str(sampling_params)
                res["model"] = model_name
                res["processed_queries"] = len(dataset)
                res["batch_duration_seconds"] = batch_duration
                res["avg_tokens_per_second"] = avg_tokens_per_sec
                res.update(monitor.get_all_metrics())
                all_results.append(res)
        
                # Save results
                pd.DataFrame(all_results).to_csv(result_folder_path+"/llm_batch_optimized_results.csv", index=False)
                pd.DataFrame(query_log).to_csv(result_folder_path+"/llm_batch_optimized_details.csv", index=False)
                
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

            # Log model failure for summary at the end
            failed_models.append({
                'model': model_name,
                'error': str(e),
                'error_type': type(e).__name__
            })

        finally:
            # Comprehensive cleanup using new utility function
            print(f"\nExecuting cleanup for model: {model_name}")
            cleanup_success = utils.cleanup_vllm_resources(
                llm=llm if llm is not None else None,
                model_name=model_name,
                verbose=True
            )

            if not cleanup_success:
                print(f"WARNING: Cleanup completed with errors for {model_name}")
                print("This may affect subsequent models. Continuing anyway...")

            # Small delay to ensure cleanup completes
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

    nvmlShutdown()

    # Force exit if any models failed to ensure no hanging processes
    # os._exit(1) with error code so bash script can handle it
    if failed_models:
        print("Forcing process exit due to model failures to prevent hanging...")
        print("Exiting with error code 1...")
        os._exit(1)