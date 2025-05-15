import collections.abc
import os, sys
import os.path as osp
from torch import nn
import kornia.augmentation as K
import pydiffvg
import save_svg
import cv2
from ttf import font_string_to_svgs, normalize_letter_size
import torch
import numpy as np
import random

from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from scr import SCR

def init_point(img, num_stroke):
    """Return init points

    Args:
        img (PIL.Image): Input Image
        num_stroke (np.array): Init points

    Returns:

    """
    img = np.array(img)
    points = np.where(img == 0.0)
    index = np.random.choice(list(range(len(points[0]))), num_stroke, replace=True)
    return points, index
def make_text_img(text, font_path):
    """return text image

    Args:
        text (string): text to convert

    Returns:
        PIL.Image : Image of text
    """
    ttfontname = font_path
    fontsize = 430
    canvasSize = (512, 512)
    backgroundRGB = (255, 255, 255)
    textRGB = (0, 0, 0)
    img = Image.new("RGB", canvasSize, backgroundRGB)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(ttfontname, fontsize)
    textWidth, textHeight = draw.textsize(text, font=font)
    textTopLeft = (
        canvasSize[0] // 2 - textWidth // 2,
        canvasSize[1] // 2 - textHeight // 2 - 70,
    )
    draw.text(textTopLeft, text, fill=textRGB, font=font)
    return img

def ensure_dir(path: str):
    """create directories if *path* does not exist"""""
    if not os.path.exists(path):
        os.mkdir(path)


def setup_logger(output=None, distributed_rank=0):
    """
    Initialize the cvpods logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        logging.Logger: a logger
    """
    logger.remove()
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # stdout logging: master only
    if distributed_rank == 0:
        logger.add(sys.stderr, format=loguru_format)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log_running.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        ensure_dir(os.path.dirname(filename))
        logger.add(filename)


def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]))
        return xnew
    else:
        return x


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


def update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def preprocess(font, word, letter, level_of_cc=1):

    if level_of_cc == 0:
        target_cp = None
    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }
        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    print(f"======= {font} =======")
    font_path = f"code/data/fonts/{font}.ttf"
    init_path = f"code/data/init"
    subdivision_thresh = None
    font_string_to_svgs(init_path, font_path, word, target_control=target_cp,
                        subdivision_thresh=subdivision_thresh)
    normalize_letter_size(init_path, font_path, word)

    # optimaize two adjacent letters
    if len(letter) > 1:
        subdivision_thresh = None
        font_string_to_svgs(init_path, font_path, letter, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(init_path, font_path, letter)

    print("Done preprocess")


def get_data_augs(cut_size):
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
    augmentations.append(K.RandomCrop(size=(cut_size, cut_size), pad_if_needed=True, padding_mode='reflect', p=1.0))
    return nn.Sequential(*augmentations)


'''pytorch adaptation of https://github.com/google/mipnerf'''
def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.
  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.
  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.
  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp


def save_image(img, filename, gamma=1):
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)


def get_letter_ids(letter, word, shape_groups):
    for group, l in zip(shape_groups, word):
        if l == letter:
            return group.shape_ids


def combine_word(word, letter, font, experiment_dir):
    word_svg_scaled = f"./code/data/init/{font}_{word}_scaled.svg"
    canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
    letter_ids = []
    for l in letter:
        letter_ids += get_letter_ids(l, word, shape_groups_word)

    w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
    h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

    c_w = (-w_min + w_max) / 2
    c_h = (-h_min + h_max) / 2

    svg_result = os.path.join(experiment_dir, "output-svg", "output.svg")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

    out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes])
    out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes])

    out_c_w = (-out_w_min + out_w_max) / 2
    out_c_h = (-out_h_min + out_h_max) / 2

    scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
    scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

    if scale_canvas_h > scale_canvas_w:
        wsize = int((out_w_max - out_w_min) * scale_canvas_h)
        scale_canvas_w = wsize / (out_w_max - out_w_min)
        shift_w = -out_c_w * scale_canvas_w + c_w
    else:
        hsize = int((out_h_max - out_h_min) * scale_canvas_w)
        scale_canvas_h = hsize / (out_h_max - out_h_min)
        shift_h = -out_c_h * scale_canvas_h + c_h

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h
        if scale_canvas_h > scale_canvas_w:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
        else:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

    for j, s in enumerate(letter_ids):
        shapes_word[s] = shapes[j]

    save_svg.save_svg(
        f"{experiment_dir}/{font}_{word}_{letter}.svg", canvas_width, canvas_height, shapes_word,
        shape_groups_word)

    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes_word, shape_groups_word)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
               torch.ones(img.shape[0], img.shape[1], 3, device="cuda:0") * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    save_image(img, f"{experiment_dir}/{font}_{word}_{letter}.png")


def init_curves(opt, init_points, index, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    group_size = 50
    num_groups = (len(index) + group_size - 1) // group_size
    for g in range(num_groups):
        group_index_start = g * group_size
        group_index_end = min((g + 1) * group_size, len(index))

        for i in range(group_index_start, group_index_end):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (
                float(init_points[1][index[i]] / canvas_width),
                float(init_points[0][index[i]] / canvas_height),
            )
            #p0 = RandomCoordInit(canvas_height, canvas_width)
            points.append(p0)
            for j in range(num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=points,
                stroke_width=torch.tensor(1.0),
                is_closed=False,
            )
            shapes.append(path)
            stroke_color_init = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,  # 无填充颜色
                stroke_color=stroke_color_init
            )
            shape_groups.append(path_group)

    return shapes, shape_groups


def create_video(num_iter, experiment_dir, video_frame_freq):
    img_array = []
    for ii in range(0, num_iter):
        if ii % video_frame_freq == 0 or ii == num_iter - 1:
            filename = os.path.join(
                experiment_dir, "video-png", f"iter{ii:04d}.png")
            img = cv2.imread(filename)
            img_array.append(img)

    video_name = os.path.join(
        experiment_dir, "video.mp4")
    check_and_create_dir(video_name)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (600, 600))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()


def make_dist(img):
    """Make distans transform map

    Args:
        img (PIL.Image): Input image

    Returns:
        torch.tensor : distans transform map
    """
    np_img = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    dist = cv2.distanceTransform(gray, cv2.DIST_L2, maskSize=0)
    cv2.normalize(dist, dist, 0, 100.0, cv2.NORM_MINMAX)
    dist = torch.from_numpy(dist).to(torch.float32)
    dist = dist.pow(1.0)
    dist = dist.to(pydiffvg.get_device())
    return dist

def calc_dist(img, dist):
    target_dist = img.clone()
    # target_dist = 255 - target_dist
    for i in range(3):
        target_dist[:, i, :, :] = target_dist[:, i, :, :] * dist

    return target_dist

def train_phi_model_refl(self,
                             pred_rgb: torch.Tensor,
                             weight: float = 1,
                             new_timesteps: bool = True):
        # interp to 512x512 to be fed into vae.
        pred_rgb_ = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode2latent(pred_rgb_)

        # get phi particles
        indices = torch.randperm(latents.size(0))
        latents_phi = latents[indices[:self.phi_n_particle]]
        latents_phi = latents_phi.detach()

        # get timestep
        if new_timesteps:
            t = torch.randint(0, self.num_train_timesteps, (1,), device=self.device)
        else:
            t = self.t

        noise = torch.randn_like(latents_phi)
        noisy_latents = self.scheduler.add_noise(latents_phi, noise, t)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents_phi, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # predict the noise residual and compute loss
        noise_pred = self.unet_phi(
            noisy_latents, t,
            encoder_hidden_states=self.text_embedd_cond,
            cross_attention_kwargs=self.lora_cross_attention_kwargs,
        ).sample

        rewards = torch.tensor(weight, dtype=torch.float32, device=self.device)
        return rewards * F.mse_loss(noise_pred, target, reduction="mean")
"""
def target_file_preprocess(self, tar_path: AnyPath):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ])

        tar_pil = Image.open(tar_path).convert("RGB")  # open file
        target_img = process_comp(tar_pil)  # preprocess
        target_img = target_img.to(self.device)
        return target_img
"""
def build_scr():
    scr = SCR(
        temperature=0.07,
        mode="refinement",
        image_size=96)
    print("Loaded SCR module for supervision successfully!")
    return scr

# Delete the thin group
def filter_shapes_by_width(shapes, shape_groups, min_width_threshold):
    filtered_shapes = []
    filtered_shape_groups = []

    for shape, shape_group in zip(shapes, shape_groups):
        # 检查路径宽度
        if shape.stroke_width >= min_width_threshold:
            print(shape.stroke_width)
            filtered_shapes.append(shape)
            filtered_shape_groups.append(shape_group)
        else:
            print(shape.stroke_width)
            # 将颜色设为透明（可选）
            shape_group.stroke_color = torch.tensor([0.0, 0.0, 0.0, 0.0])

    return filtered_shapes, filtered_shape_groups

class RandomCoordInit:
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width, self.canvas_height = canvas_width, canvas_height

    def __call__(self):
        w, h = self.canvas_width, self.canvas_height
        return [np.random.uniform(0, 1) * w, np.random.uniform(0, 1) * h]
