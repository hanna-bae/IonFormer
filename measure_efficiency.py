import time
import numpy as np
import torch
# find the meaninig of the import module 

def cuda_time() -> float:
    # waits for everything to finish running
    torch.cuda.synchronize() 
    # time include waiting delay 
    return time.perf_counter() 

def measure(model, img_size, num_repeates=500, num_warmup=500):
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

# GPU Memory consumption my code 
def to_memory(model):
    allocated = round(torch.cuda.memory_allocated(0)/1024**3, 1)
    cached = round(torch.cuda.memory_cached(0)/1024**3, 1)


# Parameter count
def model_parameter(model):
    parameter = filter(lambda  p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameter])