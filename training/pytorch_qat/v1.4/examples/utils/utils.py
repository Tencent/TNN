import os
import time
import torch
import logging

from torchvision.models.resnet import *
from torchvision.models.mobilenet import mobilenet_v2

def load_model(config):
    if config.arch == 'mobilenetv2':
        model = mobilenet_v2(pretrained=True)
    elif config.arch == 'resnet18':
        model = resnet18(pretrained=True)
    elif config.arch == 'resnet50':
        model = resnet50(pretrained=True)
    return model

def print_size_of_model(model):

    torch.save(model.state_dict(), "temp.p")

    size = os.path.getsize("temp.p")/1e6

    logging.getLogger().info(f'Size (MB):{size}')

    os.remove('temp.p')

def run_benchmark(model, img_loader):
    elapsed = 0
    model.eval()
    num_batches = 5
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    logging.getLogger().info('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))

    return elapsed

def onnx_convert(model, x = torch.randn(1, 3, 224, 224, requires_grad=True)):

    # Export the model
    torch.onnx.export(model,
                      x,
                      "output_fake_auto.onnx",
                      input_names=['input'],
                      example_outputs=['output'],
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      verbose=False, training=False, opset_version=9,
                      dynamic_axes={'input': {0: 'batch_size'},'output' : {0 : 'batch_size'}}
                      )