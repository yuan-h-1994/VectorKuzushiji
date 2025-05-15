from typing import Mapping
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from losses import ToneLoss, ConformalLoss, DistTransLoss
from losses import SDSLoss
from losses import DiscriminatorLoss
from losses import PointMinDistanceLoss
from losses import PerceptualLoss
from losses import StyleContentLoss,DistanceLoss, StrokeEncoder
from config import set_config
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word,
    create_video,
    init_curves,
    filter_shapes_by_width)
from ttf import font_string_to_beziers
import numpy as np
import wandb
import warnings
warnings.filterwarnings("ignore")
from loguru import logger
from utils import setup_logger
from xing_loss import xing_loss
import pandas as pd

from lffont.evaluator import eval_ckpt



pydiffvg.set_print_timing(False)
gamma = 1.0


def init_shapes(svg_path, trainable: Mapping[str, bool]):

    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)

    return shapes_init, shape_groups_init, parameters

@logger.catch
def main():
    cfg = set_config()
    output_dir = f"/home/Word-As-Image-Exchange-SDSLoss/Word-As-Image/output/conformal_0.5_dist_pixel_100_kernel201_{cfg.word}/NotoSansJP-VariableFont_wght/NotoSansJP-VariableFont_wght_{cfg.optimized_letter}_scaled_concept_kuzushiji_seed_0/"
    root_dir = f"/home/Word-As-Image-Exchange-SDSLoss/Word-As-Image/output/"

    if os.path.exists(os.path.join(cfg.experiment_dir, "log_running.txt")):
        os.remove(os.path.join(cfg.experiment_dir, "log_running.txt"))
    setup_logger(cfg.experiment_dir, distributed_rank=0)

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    print("preprocessing")
    preprocess(cfg.font, cfg.word, cfg.optimized_letter, cfg.level_of_cc)

    if cfg.loss.sds_loss.use_sds_loss:
        sds_loss = SDSLoss(cfg, device)
    
    discriminator_loss = DiscriminatorLoss(cfg, device)

    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(svg_path=cfg.target, trainable=cfg.trainable)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]

    if cfg.use_wandb:
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    if cfg.loss.tone.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)

    if cfg.loss.style_loss.use_style_loss:
        style_loss = StyleContentLoss(cfg, device)
        
    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)


    if cfg.loss.conformal.use_conformal_loss:
        conformal_loss = ConformalLoss(parameters, device, cfg.optimized_letter, shape_groups)
        conformal_loss.save_mesh("triangulated_mesh.png")
    
    if cfg.loss.point_min_dist.use_point_min_dist_loss:
        point_min_dist_loss = PointMinDistanceLoss(parameters, cfg=cfg)

    

    lr_lambda = lambda step: learning_rate_decay(step, cfg.lr.lr_init, cfg.lr.lr_final, num_iter,
                                                 lr_delay_steps=cfg.lr.lr_delay_steps,
                                                 lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f
    distance_loss_fn = DistanceLoss(parameters, device, cfg.optimized_letter, shape_groups)
    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    loss_dataframe = {
        "step": [],
        "loss_all": [],
        "loss_sds": [],
        "loss_discriminator": [],
        "loss_tone": [],
        "loss_point_min_dist": [],
        "loss_style": [],
        "loss_stroke": [],  
        "loss_conformal": [],  
    }

    for step in t_range:
        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]['lr']}, step=step)
        optim.zero_grad()

        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)

        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]


        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            save_image(img, os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"), gamma)
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
            check_and_create_dir(filename)
            save_svg.save_svg(
                filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = data_augs.forward(x)

        loss = 0
        ##################################################
        # discriminator loss
        ##################################################
        if cfg.loss.discriminator_loss.use_discriminator_loss:
            print("Using discriminator loss")
            loss_discriminator = discriminator_loss(x_aug)
            loss += loss_discriminator
            if loss.item() > 0 and cfg.loss.discriminator_loss.discriminator_loss_weight > 0:
                loss_dataframe["loss_discriminator"].append(loss_discriminator.item()/cfg.loss.discriminator_loss.discriminator_loss_weight)
            else:
                loss_dataframe["loss_discriminator"].append(loss_discriminator.item())
            if cfg.use_wandb:
                wandb.log({"discriminator_loss": loss}, step=step)
        else:
            loss_dataframe["loss_discriminator"].append(0)

        ##################################################
        # sds loss
        # compute diffusion loss per pixel
        ##################################################
        if cfg.loss.sds_loss.use_sds_loss:
            print("Using sds loss")
            loss_sds = sds_loss(x_aug)
            if loss_sds.item() > 0 and cfg.loss.sds_loss.sds_loss_weight > 0:
                loss_dataframe["loss_sds"].append(loss_sds.item()/cfg.loss.sds_loss.sds_loss_weight)
            else:
                loss_dataframe["loss_sds"].append(loss_sds.item())
            loss = loss + loss_sds
            if cfg.use_wandb:
                wandb.log({"sds_loss": loss_sds}, step=step)
        else:
            loss_dataframe["loss_sds"].append(0)

        ##################################################
        # tone loss
        ##################################################
        if cfg.loss.tone.use_tone_loss:
            print("Using tone loss")
            tone_loss_res = tone_loss(x, step)
            if tone_loss_res.item() > 0 and cfg.loss.tone.tone_loss_weight > 0:
                loss_dataframe["loss_tone"].append(tone_loss_res.item()/cfg.loss.tone.tone_loss_weight)
            else:
                loss_dataframe["loss_tone"].append(tone_loss_res.item())
            if cfg.use_wandb:
                wandb.log({"tone_loss": tone_loss_res}, step=step)
            loss = loss + tone_loss_res
        else:
            loss_dataframe["loss_tone"].append(0)

        ##################################################
        # conformal
        ##################################################
        if cfg.loss.conformal.use_conformal_loss:
            print("Using conformal loss")
            loss_angles = conformal_loss()
            if loss_angles.item() > 0 and cfg.loss.conformal.angeles_w > 0:
                loss_dataframe["loss_conformal"].append(loss_angles.item()/cfg.loss.conformal.angeles_w)
            else:
                loss_dataframe["loss_conformal"].append(loss_angles.item())
            if cfg.use_wandb:
                wandb.log({"loss_angles": loss_angles}, step=step)
            loss = loss + loss_angles
        else:
            loss_dataframe["loss_conformal"].append(0)
        
        ##################################################
        # point_min_dist
        ##################################################
        if cfg.loss.point_min_dist.use_point_min_dist_loss:
            print("Using point_min loss")
            point_min_dist_loss_res, masked_num = point_min_dist_loss()
            if point_min_dist_loss_res.item() > 0 and cfg.loss.point_min_dist.point_min_dist_loss_weight > 0:
                loss_dataframe["loss_point_min_dist"].append(point_min_dist_loss_res.item()/cfg.loss.point_min_dist.point_min_dist_loss_weight)
            else:
                loss_dataframe["loss_point_min_dist"].append(point_min_dist_loss_res.item())
            if cfg.use_wandb:
                wandb.log({"loss_point_min_dist": point_min_dist_loss_res}, step=step)
                wandb.log({"masked_num": masked_num}, step=step)
            loss = loss + point_min_dist_loss_res
        else:
            loss_dataframe["loss_point_min_dist"].append(0)

        ##################################################
        # sls_loss
        ##################################################
        # Change to the content image~~~
        if cfg.loss.sls_loss.use_sls_loss:
            print("Using sls loss")
            config_path = "code/mxencoder/mxfont/cfgs/defaults.yaml"
            weight_path = "code/mxencoder/mxfont/generator.pth"
            stroke_encoder = StrokeEncoder(config_path, weight_path, device)
            loss_stroke = stroke_encoder.feature_loss(x_aug)
            if loss_stroke.item() > 0 and(cfg.loss.sls_loss.sls_loss_weight > 0):
                loss_dataframe["loss_stroke"].append(loss_stroke.item()/cfg.loss.sls_loss.sls_loss_weight)
            else:
                loss_dataframe["loss_stroke"].append(loss_stroke.item())
            loss = loss + loss_stroke

        ##################################################
        # style_loss
        ##################################################
        if cfg.loss.style_loss.use_style_loss:
            style_loss_res = style_loss(x_aug)
            if style_loss_res.item() > 0 and cfg.loss.style_loss.style_loss_weight > 0:
                loss_dataframe["loss_style"].append(style_loss_res.item()/cfg.loss.style_loss.style_loss_weight)
            else:
                loss_dataframe["loss_style"].append(style_loss_res.item())
            if cfg.use_wandb:
                wandb.log({"style_loss": style_loss_res}, step=step)
            loss = loss + style_loss_res
        else:
            loss_dataframe["loss_style"].append(0)

        if cfg.use_wandb:
            wandb.log({"loss_all": loss}, step=step)

        loss_all = loss.item()
        loss_info = f"step {step}, loss: {loss_all}"
        logger.info(loss_info)

        # add to loss dataframe
        loss_dataframe["step"].append(step)
        loss_all_unscale = 0
        for loss_i in loss_dataframe:
            if loss_i != "step" and loss_i != "loss_all":
                loss_all_unscale += loss_dataframe[loss_i][-1]
        loss_dataframe["loss_all"].append(loss_all_unscale)

        t_range.set_postfix({'loss': loss_all})
        loss.backward()
        optim.step()
        scheduler.step()

    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)

    if cfg.save.image:
        filename = os.path.join(
            cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        imshow = img.detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()
    return_img_list = []
    if cfg.num_iter >= 50:
        return_img_list.append(os.path.join(output_dir, f"video-png/iter0049.png"))
    else:
        return_img_list.append(os.path.join(root_dir, "none.png"))
    for i in range(100, 501, 100):
        idx = i - 1
        if idx < cfg.num_iter:
            return_img_list.append(os.path.join(output_dir, f"video-png/iter{idx:04d}.png"))
        else:
            return_img_list.append(os.path.join(root_dir, "none.png"))
    
    # convert loss list to pandas dataframe
    loss_dataframe_pd = pd.DataFrame(loss_dataframe)
    
    loss_text = f"loss_all: {loss_dataframe_pd['loss_all'].iloc[-1]:.2f}\n" + \
        f"loss_sds: {loss_dataframe_pd['loss_sds'].iloc[-1]:.2f}\n" + \
        f"loss_discriminator: {loss_dataframe_pd['loss_discriminator'].iloc[-1]:.2f}\n" + \
        f"loss_tone: {loss_dataframe_pd['loss_tone'].iloc[-1]:.2f}\n" + \
        f"loss_conformal: {loss_dataframe_pd['loss_conformal'].iloc[-1]:.2f}\n" + \
        f"loss_point_min_dist: {loss_dataframe_pd['loss_point_min_dist'].iloc[-1]:.2f}\n" + \
        f"loss_style: {loss_dataframe_pd['loss_style'].iloc[-1]:.2f}"
    
    return os.path.join(output_dir, "video.mp4"), *return_img_list, loss_text, 

if __name__ == "__main__":

    main()