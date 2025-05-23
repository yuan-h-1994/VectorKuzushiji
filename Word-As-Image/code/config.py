import argparse
import os.path as osp
import yaml
import random
from easydict import EasyDict as edict
import numpy.random as npr
import torch
from utils import (
    edict_2_dict,
    check_and_create_dir,
    update)
import wandb
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config/base.yaml")
    parser.add_argument("--experiment", type=str, default="conformal_0.5_dist_pixel_100_kernel201")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--log_dir', metavar='DIR', default="output")
    parser.add_argument('--font', type=str, default="NotoSansJP-VariableFont_wght", help="font name")
    parser.add_argument('--semantic_concept', type=str, default="Japanese Kuzushiji", help="the semantic concept to insert")
    #parser.add_argument('--word', type=str, default="none", help="the text to work on")
    parser.add_argument('--prompt_suffix', type=str, default="minimal flat 2d vector. lineal color."
                                                             "trending on artstation")
                                                             #"Japanese font inspired by classical calligraphy, with fluid strokes and natural imperfections")
    #parser.add_argument('--optimized_letter', type=str, default="none", help="the letter in the word to optimize")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--wandb_user', type=str, default="none")

    parser.add_argument('--num_iter', type=int, default=300, help='Number of iterations.')
    parser.add_argument('--word', type=str, default='今日', help='The word to process.')
    parser.add_argument('--optimized_letter', type=str, default='今', help='The letter to optimize.')
    parser.add_argument('--use_sds_loss',  help='Whether to use SDS loss. Default is True.')
    parser.add_argument('--sds_loss_weight', type=float, default=1.0, help='Weight of the SDS loss.')
    parser.add_argument('--use_discriminator_loss',  help='Whether to use discriminator loss. Default is True.')
    parser.add_argument('--discriminator_loss_weight', type=float, default=100, help='Weight of the discriminator loss.')
    parser.add_argument('--use_tone_loss',  help='Whether to use tone loss. Default is True.')
    parser.add_argument('--tone_loss_weight', type=float, default=1, help='Weight of the tone loss.')
    parser.add_argument('--use_dist_trans_loss',  help='Whether to use distance transform loss. Default is True.')
    parser.add_argument('--dist_trans_loss_weight', type=float, default=100, help='Weight of the distance transform loss.')
    parser.add_argument('--use_conformal_loss',  help='Whether to use conformal loss. Default is True.')
    parser.add_argument('--conformal_loss_angeles_w', type=float, default=0.5, help='Weight of the conformal loss.')
    parser.add_argument('--use_point_min_dist_loss',  help='Whether to use point minimum distance loss. Default is True.')
    parser.add_argument('--point_min_dist_loss_weight', type=float, default=0.01, help='Weight of the point minimum distance loss.')
    parser.add_argument('--point_min_dist_loss_distance_c', type=float, default=5, help='Coefficient for the point minimum distance loss.')
    parser.add_argument('--use_perceptual_loss',  help='Whether to use perceptual loss. Default is True.')
    parser.add_argument('--perceptual_loss_weight', type=float, default=150, help='Weight of the perceptual loss.')
    parser.add_argument('--use_style_loss',  help='Whether to use style loss. Default is True.')
    parser.add_argument('--style_loss_weight', type=float, default=500, help='Weight of the style loss.')
    parser.add_argument('--content_weight', type=float, default=100, help='Weight of the content loss.')


    cfg = edict()
    args = parser.parse_args()
    with open('TOKEN', 'r') as f:
        setattr(args, 'token', f.read().replace('\n', ''))
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.font = args.font
    cfg.semantic_concept = args.semantic_concept
    cfg.word = cfg.semantic_concept if args.word == "none" else args.word
    if " " in cfg.word:
      raise ValueError(f'no spaces are allowed')
    cfg.caption = f"a {args.semantic_concept}. {args.prompt_suffix}"
    cfg.log_dir = f"{args.log_dir}/{args.experiment}_{cfg.word}"
    if args.optimized_letter in cfg.word:
        cfg.optimized_letter = args.optimized_letter
    else:
      raise ValueError(f'letter should be in word')
    cfg.batch_size = args.batch_size
    cfg.token = args.token
    cfg.use_wandb = args.use_wandb
    cfg.wandb_user = args.wandb_user
    cfg.letter = f"{args.font}_{args.optimized_letter}_scaled"
    cfg.target = f"code/data/init/{cfg.letter}"

    cfg.num_iter = args.num_iter
    cfg.optimized_letter = args.optimized_letter
    cfg.tone_loss_weight = args.tone_loss_weight

    return cfg


def set_config():

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg_full = yaml.load(f, Loader=yaml.FullLoader)

    # recursively traverse parent_config pointers in the config dicts
    cfg_key = cfg_arg.experiment
    cfgs = [cfg_arg]
    while cfg_key:
        cfgs.append(cfg_full[cfg_key])
        cfg_key = cfgs[-1].get('parent_config', 'baseline')

    # allowing children configs to override their parents
    cfg = edict()
    for options in reversed(cfgs):
        update(cfg, options)
    del cfgs

    # set experiment dir
    signature = f"{cfg.letter}_concept_{cfg.semantic_concept}_seed_{cfg.seed}"
    cfg.experiment_dir = \
        osp.join(cfg.log_dir, cfg.font, signature)
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    print('Config:', cfg)

    # create experiment dir and save config
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    if cfg.use_wandb:
        wandb.init(project="Word-As-Image", entity=cfg.wandb_user,
                   config=cfg, name=f"{signature}", id=wandb.util.generate_id())

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = False
    else:
        assert False

    return cfg