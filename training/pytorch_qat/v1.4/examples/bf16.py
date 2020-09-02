import sys
import os
import time
import argparse
import logging
import warnings

import torch
from torch import nn
import torchvision.models as models
import torch.backends.cudnn as cudnn

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from quantization_wrap.quantization.fake_bf16_qconfig import default_fake_bf16_qconfig
from quantization_wrap.quant_model import QuantizableModel

from utils.metrics import AverageMeter, accuracy, evaluate
from utils.utils import load_model
from utils.config import process_config

from datasets.imagenet import Imagenet

warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append('shufflenet')
model_names.append('mobilenetv2')

parser = argparse.ArgumentParser(description=' ')

# configuration
parser.add_argument('--config_mode', default=1,
        help='Preload configuration mode selection')

parser.add_argument('--config_path', default='configs/qat_mobilenetv2.json',
        help='The Configuration file in json format')

# expriment name
parser.add_argument('--exp_name', default='bf16')

# quantization scheme
parser.add_argument('-qs', default='qat',  help='quantization scheme ("pq" or "qat")')

# data loader configs
parser.add_argument('--data_path', metavar='DIR', default='./data/',
                    help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=96)
parser.add_argument('--test_batch_size', type=int, default=96)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--pin_memory', default=True)

# train configs
parser.add_argument('--train_print', type=int, default=100)
parser.add_argument('--eval_print', type=int, default=200000)

# model configs
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--max_epoch', default=30, type=int)

parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenetv2',
                    choices=model_names,
                    help = 'model architecture: {model_names}  (default: mobilenetv2)'.format(model_names=model_names))
                    
parser.add_argument('--seed', default=191009, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--device', default='cpu',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device_ids', default=[2], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')

config = process_config(parser.parse_args())

def train_one_epoch(nepoch, model, criterion, optimizer, data_imagenet, device):

    model.train()

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    
    iter_num = 0

    strat_time = time.time()

    epoch_time = time.time()

    for image, target in data_imagenet.data_loader:

        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))

        if iter_num % config.train_print == 0:
            
            logging.getLogger().info('Epoch: {nepoch}, Step {step}, train accuracy: Acc@1: {top1.avg:.3f} Acc@5: {top5.avg:.3f}, loss: {loss.avg:.3f}, time:{time:.1f}s'
                .format(nepoch=nepoch, step=iter_num, top1=top1, top5=top5, loss=avgloss, time=time.time()-strat_time))
            
            strat_time = time.time()

        if iter_num == 0: 
            iter_num = iter_num + 1
            continue    

        if iter_num % config.eval_print == 0:

            if nepoch > 3:
                model.apply(torch.quantization.disable_observer)
            if nepoch > 2:
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            evaluate_start = time.time()
            top1, _ = evaluate(model, data_imagenet.data_loader_test, device)
            logging.getLogger().info('Epoch: {nepoch}, evaluation accuracy: {top1.avg:2.2f}; itme, {time:.1f} s'.format(nepoch=nepoch, top1=top1,  time=time.time()-evaluate_start))
            
            model.train()

        iter_num = iter_num + 1

    logging.getLogger().info('Epoch: {nepoch}, train accuracy: Acc@1: {top1.avg:.3f} Acc@5: {top5.avg:.3f}, loss: {loss.avg:.3f}, time:{time:.1f}s'
                .format(nepoch=nepoch, step=iter_num, top1=top1, top5=top5, loss=avgloss, time=time.time()-epoch_time))

    return

def main():

    torch.manual_seed(config.seed)

    saved_model_dir = config.out_dir

    data_imagenet= Imagenet(config)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    device = torch.device(config.device)

    logging.getLogger().info('{config.arch} {config2.qs} start.............'.format(config=config, config2=config))

    qat_model = load_model(config)
    x = torch.ones(1, 3, 224, 224)
    qat_model = QuantizableModel(qat_model, [1, 3, 224, 224], default_fake_bf16_qconfig)

    if 'cuda' in config.device and torch.cuda.is_available():
        device = torch.device(config.device)
        if config.seed is not None:
            torch.cuda.manual_seed_all(config.seed)
        torch.cuda.set_device(config.device_ids[0])
        cudnn.benchmark = True
        logging.getLogger().info("Program will run on *****GPU-CUDA***** ")
    else:
        config.device_ids = None
        device = torch.device('cpu')
        logging.getLogger().info("Program will run on *****CPU***** ")

    qat_model.to(device)

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=config.learning_rate)

    if 'pq' in config.qs: config.max_epoch = 1

    for nepoch in range(config.max_epoch):

        train_one_epoch(nepoch, qat_model, criterion, optimizer, data_imagenet, device)

        if nepoch > 3:
            qat_model.apply(torch.quantization.disable_observer)
        if nepoch > 2:
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        strat_time = time.time()
        top1, _ = evaluate(qat_model, data_imagenet.data_loader_test, device)
        logging.getLogger().info('Epoch: %d, evaluation accuracy, %2.2f, Time: %2.2f s' % (nepoch, top1.avg, time.time() - strat_time))

        torch.save(qat_model.state_dict(), f'{saved_model_dir}/best.pth'.format(path=saved_model_dir))

if __name__ == "__main__":
    main()


