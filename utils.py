from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown, nvmlDeviceGetTemperature
import time
import threading
import psutil

class MonitorThread(threading.Thread):
    def __init__(self, handle, secs_between_samples=1):
        super().__init__(daemon=True)
        self.handle = handle
        self.secs_between_samples = secs_between_samples
        self.gpu_utilization = []
        self.gpu_mem_utilization = []
        self.cpu_utilization = []
        self.ram_utilization = []
        self.running = True

    def run(self):
        while self.running:
            self.gpu_mem_utilization.append(nvmlDeviceGetMemoryInfo(self.handle).used / (1024 * 1024))
            self.gpu_utilization.append(nvmlDeviceGetUtilizationRates(self.handle).gpu)
            self.ram_utilization.append(psutil.virtual_memory().used / (1024 * 1024))
            self.cpu_utilization.append(psutil.cpu_percent())
            time.sleep(self.secs_between_samples)

    def stop(self):
        self.running = False

    def get_all_metrics(self):
        all_metrics = {}
        all_metrics["gpu_memory_used_mb"] = self.gpu_mem_utilization
        all_metrics["gpu_utilization_percent"] = self.gpu_utilization
        all_metrics["cpu_memory_used_mb"] = self.ram_utilization
        all_metrics["cpu_utilization_percent"] = self.cpu_utilization

        return all_metrics

def wait_for_gpu_cooldown(handle, target_temp=50, check_interval=5):
    """Wait until the GPU temperature drops below the target temperature."""
    gpu_temp = nvmlDeviceGetTemperature(handle, 0)  # 0 -> NVML_TEMPERATURE_GPU
    print(f"GPU temperature before cooldown: {gpu_temp}°C")

    while gpu_temp > target_temp:
        print(f"Waiting for GPU to cool down... Current temp: {gpu_temp}°C")
        time.sleep(check_interval)
        gpu_temp = nvmlDeviceGetTemperature(handle, 0)

    print(f"GPU cooled down to {gpu_temp}°C. Continue.")

