from dataset.dataloader import build_dataset
from models import build_model
import util.misc as utils

import argparse
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('Setting', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--resume', default='./exps/bs8_epo300_sgdlr1e-3.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    
    # dataset parameters
    parser.add_argument('--dataset_path', default='/host/space0/chen-j/HowToEat', type=str)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HowToEat face classfication training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    save_fig = True

    device = torch.device(args.device)

    model, criterion = build_model(args)
    model.to(device)

    dataset_train = build_dataset(args, "train")
    dataset_val = build_dataset(args, "test")

    print(len(dataset_train), len(dataset_val))


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    model.eval()
    criterion.eval()

    count = 0
    right = 0

    eating = 0
    not_eating = 0

    for sample_i, img_i, target_i in tqdm(dataset_val):
        sample = sample_i.unsqueeze(0).to(device)
        target = target_i.unsqueeze(0).to(device)
        
        output = model(sample)
        loss_dict = criterion(output, target)

        probas = output["pred_logits"].softmax(-1)

        count += 1

        if torch.argmax(probas[0]) ==  target[0]:
            right += 1
        
            if torch.argmax(probas[0]) == 0:

                if eating < 10 and save_fig:
                    plt.imshow(img_i)
                    # plt.title(str(probas.cpu().detach_().numpy()[0].tolist()))
                    plt.axis('off')

                    plt.savefig(f"test_eating_{eating}.png",
                            format="png",
                            dpi=175,
                            bbox_inches='tight')
                    eating += 1
            else:
                if not_eating < 10 and save_fig:
                    plt.imshow(img_i)
                    # plt.title(str(probas.cpu().detach_().numpy()[0].tolist()))
                    plt.axis('off')

                    plt.savefig(f"test_not_eating_{not_eating}.png",
                            format="png",
                            dpi=175,
                            bbox_inches='tight')
                    not_eating += 1

        if eating >= 10 and not_eating >= 10 and save_fig:
            break
    print(right/count)

    print("finish")
