from dataset.dataloader import build_dataset
from models import build_model
from engine import train_one_epoch, evaluate
import util.misc as utils

import argparse
import datetime
import time
from tqdm import tqdm
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import random


def get_args_parser():
    parser = argparse.ArgumentParser('Setting', add_help=False)
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    
    # dataset parameters
    parser.add_argument('--dataset_path', default='../data/', type=str)
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--output_dir', default='./exps',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    lowest_error = 100
    epoch_best = 0
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset_train = build_dataset(args, "train")
    dataset_val = build_dataset(args, "test")

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, criterion, data_loader_val, device, args.output_dir)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        
        test_stats = evaluate(
            model, criterion, data_loader_val, device, args.output_dir
        )

        if args.output_dir and test_stats["class_error"] < lowest_error:
            lowest_error = test_stats["class_error"]
            epoch_best = epoch
            checkpoint_paths = [output_dir / 'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'n_parameters': n_parameters,
                     'epoch': epoch,
                     'best_epoch': epoch_best}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('classfication training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
