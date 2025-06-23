import time
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
import threading

import joblib
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
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
ds = pd.read_csv("tabular/loan-prediction-subsample.csv", index_col=0).drop(['Id', 'Risk_Flag'], axis=1)
# Increase dataset size to 1M samples, otherwise inference is too fast.
ds = pd.concat([ds for _ in range(100)])

# Load ColumnTransformer.
ct = joblib.load('tabular/column_transformer.pkl')
# Load model.
model = xgb.Booster()
model.load_model("tabular/loans.model")

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
                        label="Batch Tabular", include_idle=True)

    processed_queries = 0
    total_generated_tokens = 0

    # Start monitoring
    monitor = utils.MonitorThread(handle)
    monitor.start()

    # Start energy measurement
    start_time = time.time()
    meter.begin()

    # Process all queries
    X_scaled = ct.transform(ds)
    d = xgb.DMatrix(X_scaled)
    # Predict.
    outputs = (model.predict(d) > 0.3).astype(int)
        
    print(f"Queries processed: {len(outputs)}, Total positives: {sum(outputs)}\n")

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
    res["model"] = "xgboost"
    res["processed_queries"] = len(ds)
    res.update(monitor.get_all_metrics())
    results.append(res)

    # Save results
    pd.DataFrame(results).to_csv("results/tabular_batch_results.csv")
    utils.wait_for_gpu_cooldown(handle)

nvmlShutdown()
