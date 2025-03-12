from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown, nvmlDeviceGetTemperature
import time

def wait_for_gpu_cooldown(handle, target_temp=50, check_interval=5):
    """Wait until the GPU temperature drops below the target temperature."""
    gpu_temp = nvmlDeviceGetTemperature(handle, 0)  # 0 -> NVML_TEMPERATURE_GPU
    print(f"GPU temperature before cooldown: {gpu_temp}°C")

    while gpu_temp > target_temp:
        print(f"Waiting for GPU to cool down... Current temp: {gpu_temp}°C")
        time.sleep(check_interval)
        gpu_temp = nvmlDeviceGetTemperature(handle, 0)

    print(f"GPU cooled down to {gpu_temp}°C. Continue.")

