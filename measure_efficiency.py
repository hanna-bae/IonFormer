import time
import numpy as np
import torch
import psutil

def cuda_time() -> float:
    '''Returns the current time after CUDA synchronization'''
    torch.cuda.synchronize() 
    return time.perf_counter() 

def measure(model, img_size, num_repeates=500, num_warmup=500):
    '''Measures the latency of a model given an input size.'''
    model.cuda()
    model.eval()

    backbone = model.backbone
    inputs = torch.randn(4, 3, img_size, img_size).cuda()

    latencies = []
    for k in range(num_repeates+num_warmup):
        start = cuda_time()
        backbone(inputs)
        if k >= num_warmup:
            latencies.append((cuda_time() - start)*1000)
    
    latencies = sorted(latencies)

    # remove the outlier
    drop = int(len(latencies) * 0.25)
    return np.mean(latencies[drop:-drop])


def memory_consumption(model):
    '''Returns the allocated and cached GPU memory in GB'''
    allocated = round(torch.cuda.memory_allocated(0)/1024**3, 1)
    cached = round(torch.cuda.memory_cached(0)/1024**3, 1)
    return allocated, cached


def model_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# need to modify
def measure_power_usage():
    for i in range(10):
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        print(f"CPU: {cpu_percent}% | Memory: {mem}%")
        time.sleep(1)

'''
Example Usage:
    latency = measure_latency(model, img_size=240)
    allocated_memory, cached_memory = memory_consumption(model)
    num_params = model_parameter(model)
    measure_power_usage()
'''