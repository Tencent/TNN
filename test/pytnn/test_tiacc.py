import os
import time
import torch
import torchvision.models as models
import pytnn

model = models.resnet50().float().cuda()
model = torch.jit.script(model).eval()
dummy = torch.rand(1, 3, 224, 224).cuda()

def benchmark(model, inp):
    for i in range(100):
        model(inp)
    start = time.time()
    for i in range(200):
        model(inp)
    elapsed_ms = (time.time() - start) * 1000
    print("Latency: {:.2f}".format(elapsed_ms / 200))

benchmark(model, dummy)

optimized_model, report = pytnn.optimize(model, 0)

benchmark(optimized_model, dummy)

print(report)
