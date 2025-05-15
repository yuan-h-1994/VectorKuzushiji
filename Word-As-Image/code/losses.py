import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import utils
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline

import clip

from resnet import build as build_discriminator

from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import torchvision.transforms as transforms
import random
from mxencoder.mxfont.models import Generator
from sconf import Config
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from pathlib import Path
from lffont.utils import Logger
from lffont.datasets import load_lmdb, read_data_from_lmdb
from lffont.evaluator import Evaluator


class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(cfg.diffusion.model,
                                                       torch_dtype=torch.float16, use_auth_token=cfg.token)
        self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.sds_loss_weight = cfg.loss.sds_loss.sds_loss_weight

        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(self.cfg.caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.cfg.batch_size, 0)
        del self.pipe.tokenizer
        del self.pipe.text_encoder


    def forward(self, x_aug):
        sds_loss = 0

        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, self.cfg.diffusion.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
            grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=1e5, neginf=-1e5)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()

        return sds_loss * self.sds_loss_weight


class DiscriminatorConfig:
    def __init__(self, device):
        self.device = device


class DiscriminatorLoss(nn.Module):
    def __init__(self, cfg, device):
        super(DiscriminatorLoss, self).__init__()
        self.cfg = cfg
        self.device = device

        self.pipe, self.criterion = build_discriminator(DiscriminatorConfig(device))
        self.pipe = self.pipe.to(self.device)

        print("load pretrained discriminator...")
        state_dict = torch.load("../discriminator/exps/checkpoint_best.pth", map_location=self.device)
        self.pipe.load_state_dict(state_dict['model'])

        for param in self.pipe.parameters():
            param.requires_grad = False
        
        self.loss_all = []

    def forward(self, x_aug):
        d_loss = 0

        # encode rendered image
        x = x_aug * 2. - 1.

        result = self.pipe(x)

        # result["pred_logits"] = torch.softmax(result["pred_logits"], dim=1)

        target = torch.ones_like(result["pred_logits"])
        target[:, 1] = 0

        # print(target)

        d_loss = self.criterion(result, target)

        return d_loss["loss_ce"]

"""
class PerceptualLoss(nn.Module):
    def __init__(self, cfg, device):
        super(PerceptualLoss, self).__init__()
        self.cfg = cfg
        self.device = device

        self.loss_weight = cfg.loss.perceptual_loss.perceptual_loss_weight

        self.pipe, _ = clip.load("ViT-B/32", device, jit=False)
        self.pipe = self.pipe.to(self.device)
        
        self.loss_all = []
    
    def set_image_init(self, im_init):
        with torch.no_grad():
            im_init_resize = F.interpolate(im_init.permute(2, 0, 1).unsqueeze(0), size=224, mode="bicubic")
            source_features = self.pipe.encode_image(im_init_resize)
            self.source_features = source_features / source_features.clone().norm(dim=-1, keepdim=True)

    def forward(self, x_aug):

        # encode rendered image
        x = x_aug * 2. - 1.

        x = F.interpolate(x, size=224, mode="bicubic")

        batch_features = self.pipe.encode_image(x)
        batch_features /= batch_features.clone().norm(dim=-1, keepdim=True)
        
        # l1 loss
        loss_l1 = F.l1_loss(batch_features, self.source_features)

        return loss_l1 * self.loss_weight
""" 

class PerceptualLoss(nn.Module):
    def __init__(self, cfg, device):
        super(PerceptualLoss, self).__init__()
        self.cfg = cfg
        self.device = device

        self.loss_weight = cfg.loss.perceptual_loss.perceptual_loss_weight

        self.vgg = vgg19(pretrained=True).features
        self.vgg = self.vgg.eval()

        self.target_img_path = cfg.loss.style_loss.target_img_path
        self.style_img = self.loader(self.target_img_path)
        self.style_features = self.vgg(self.style_img)

        self.loss_all = []
    
    
    def loader(self, image_name):
        loader = transforms.Compose([
            transforms.Resize((64, 64)),  # 修改为适当的大小
            transforms.ToTensor()
        ])
        image = Image.open(image_name).convert("RGB")  # 转换为 RGB
        image = loader(image).unsqueeze(0)
        return image.to(torch.float)

    def forward(self, x_aug):

        # encode rendered image
        x = x_aug * 2. - 1.

        x = F.interpolate(x, size=224, mode="bicubic")

        batch_features = self.vgg(x)
        batch_features /= batch_features.clone().norm(dim=-1, keepdim=True)
        
        # mse loss
        loss_mse = F.mse_loss(batch_features, self.style_features)

        return loss_mse * self.loss_weight

class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.tone_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur,
                                                                        cfg.loss.tone.pixel_dist_kernel_blur), sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)

    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)


class DistTransLoss(nn.Module):
    def __init__(self, cfg):
        super(DistTransLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.dist_trans.dist_trans_loss_weight

    def set_image_init(self, im_init):
        self.im_init = utils.make_dist(im_init)
    
    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        target_dist = utils.calc_dist(cur_raster, self.im_init)
        shape_loss = (self.im_init - target_dist).pow(2).mean()
        return shape_loss


class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()
    def save_mesh(self, save_path="mesh.png", invert_axes=True, bg_color="white", edge_color="blue"):

        points = [p.clone().detach().cpu().numpy() for p in self.parameters.point]
        points = np.concatenate(points)

        if not self.faces:
            raise ValueError("self.faces is empty. Ensure faces are properly initialized before saving the mesh.")

        fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
        patches = []

        for face_group in self.faces:

            triangles = face_group.cpu().numpy()


            for tri in triangles:
                triangle_points = points[tri]
                polygon = MplPolygon(triangle_points, closed=True, edgecolor=edge_color, fill=False)
                patches.append(polygon)

        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)


        ax.set_xlim(points[:, 0].min() - 10, points[:, 0].max() + 10)
        ax.set_ylim(points[:, 1].min() - 10, points[:, 1].max() + 10)

        points[:, 1] = -points[:, 1]

        if invert_axes:
            ax.invert_xaxis()  
            ax.invert_yaxis()  

        ax.set_aspect('equal')
        ax.axis('off')
        plt.savefig(save_path, dpi=300)

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    
    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)
    ''' 
    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_
    '''
    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print("stroke check",c, start_ind, end_ind, shapes_per_letter)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            polygons = []
            for i in range(start_ind, end_ind + 1):
                pen_boundary = points_np[i]
                if not np.array_equal(pen_boundary[0], pen_boundary[-1]):  
                    pen_boundary = np.append(pen_boundary, [pen_boundary[0]], axis=0) 
                polygons.append(Polygon(pen_boundary))
            poly = MultiPolygon(polygons)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces])
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_
       
    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles


class PointMinDistanceLoss:
    def __init__(self, parameters: EasyDict, cfg):
        self.parameters = parameters
        self.point_dist_loss_w = cfg.loss.point_min_dist.point_min_dist_loss_weight
        self.C = cfg.loss.point_min_dist.point_min_dist_loss_distance_c

    def __call__(self) -> torch.Tensor:
        points = torch.cat(self.parameters.point)  # [num_points, 2]
        N, _ = points.shape
        dist_sum = torch.tensor(0.0, device=points.device)
        loss_mask_total = torch.tensor([], dtype=torch.bool, device=points.device)

        for i in range(N):
            indices = list(range(max(0, i-5), i)) + list(range(i+1, min(N, i+6)))
            if indices: 
                neighbors = points[indices] 
                dist = torch.norm(points[i] - neighbors, dim=1) - self.C 

                loss_mask = dist < 0 
                loss_mask_total = torch.cat([loss_mask_total, loss_mask])

                dist_sum += -dist[loss_mask].sum()

        masked_num = torch.sum(loss_mask_total)
        return self.point_dist_loss_w * dist_sum, masked_num


def image_loader(image_name):
    imsize = 512
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class StyleContentLoss(nn.Module):

    def __init__(self, cfg, device):
        super(StyleContentLoss, self).__init__()
        self.target_img_path = cfg.loss.style_loss.target_img_path
        self.style_weight = cfg.loss.style_loss.style_loss_weight
        self.content_weight = cfg.loss.style_loss.content_weight
        
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        images = []
        for img_path in self.target_img_path:
            style_img = image_loader(img_path)
            style_img = ((style_img - cnn_normalization_mean) / cnn_normalization_std).to(device)
            images.append(style_img)

        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        model = nn.Sequential()

        style_losses = []
        content_losses = []
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False).to(device)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            
            if name in style_layers:
                # add style loss:
                features = []
                for style_img in images:
                    target_feature = model(style_img).detach().to(device)
                    features.append(target_feature)
                average_feature = torch.mean(torch.stack(features), dim=0)
                style_loss = StyleLoss(average_feature).to(device)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        model.eval()
        model.requires_grad_(False)

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

    def set_style_image(self, style_img):
        pass

    def forward(self, input):
        self.model(input)
        style_score = 0
        content_score = 0
        for sl in self.style_losses:
            style_score += sl.loss
        for cl in self.content_losses:
            content_score += cl.loss
        style_score *= self.style_weight
        content_score *= self.content_weight
        return style_score + content_score
    
class DistanceLoss(nn.Module):
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]
        self.device = device
        self.min_threshold = 7

        with torch.no_grad():
            self.distanceL = []
            self.reset()

    def get_distanceL(self, points: torch.Tensor) -> torch.Tensor:
        distanceL_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            distanceL = length
            distanceL_.append(distanceL)
        return distanceL_
    
    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.distanceL = self.get_distanceL(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print("stroke check",c, start_ind, end_ind, shapes_per_letter)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            polygons = []
            for i in range(start_ind, end_ind + 1):
                pen_boundary = points_np[i]
                if not np.array_equal(pen_boundary[0], pen_boundary[-1]):
                    pen_boundary = np.append(pen_boundary, [pen_boundary[0]], axis=0) 
                polygons.append(Polygon(pen_boundary))

            poly = MultiPolygon(polygons)

            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces])
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_distanceL = 0
        loss_min_threshold = 0
        points = torch.cat(self.parameters.point)
        distanceL = self.get_distanceL(points)
        for i in range(len(self.faces)):
            current_distances = distanceL[i].flatten()
            num_distances = len(current_distances)
            if num_distances < 10:
                continue
            for j in range(num_distances - 5): 
                avg_prev = current_distances[max(0, j - 4):j + 1].mean()
                avg_next = current_distances[j + 1:min(num_distances, j + 6)].mean()
                diff = avg_next - avg_prev
                loss_distanceL += torch.exp(torch.abs(diff)).to(self.device)
            below_threshold = current_distances[current_distances < self.min_threshold]
            if len(below_threshold) > 0:
                loss_min_threshold += torch.sum((self.min_threshold - below_threshold) ** 2).to(self.device)

        return loss_distanceL + loss_min_threshold

class StrokeEncoder(nn.Module):
    def __init__(self, config_path: str, weight_path: str, device: torch.device):
        super(StrokeEncoder, self).__init__()
        self.device = device

        cfgs = Config(config_path)
        g_kwargs = cfgs.get('g_args', {})
        self.generator = Generator(1, 32, 1, **g_kwargs).to(device)
        self._load_weights(weight_path)

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def _load_weights(self, weight_path: str):
        weight = torch.load(weight_path, map_location=self.device)
        if "generator_ema" in weight:
            weight = weight["generator_ema"]
        self.generator.load_state_dict(weight)

    def encode(self, char_imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.generator.gen_from_style_char(char_imgs)
        
    def transform_image1(self, image_tensor):

        r, g, b = image_tensor[:, 0:1, :, :], image_tensor[:, 1:2, :, :], image_tensor[:, 2:3, :, :]
        gray_image = 0.299 * r + 0.587 * g + 0.114 * b 

        resized_image = F.interpolate(gray_image, size=(128, 128), mode="bilinear", align_corners=False)

        return resized_image

    def feature_loss(self, generated_image: torch.Tensor) -> torch.Tensor:
        target_image = image_loader("../data/melody/11.png")
        stroke_g = F.interpolate(target_image, size=(128, 128), mode='bilinear', align_corners=False)
        stroke_g = stroke_g.unsqueeze(2).to(self.device)
        stroke_t = F.interpolate(generated_image, size=(128, 128), mode='bilinear', align_corners=False)
        stroke_t = stroke_t.unsqueeze(2).to(self.device)
        target_features = self.encode(stroke_g)
        generated_features = self.encode(stroke_t)
        generated_tensor = generated_features['last']
        target_tensor = target_features['last']

        loss = F.mse_loss(generated_tensor, target_tensor)
        return loss

class EvaluatorWrapper(nn.Module):

    def __init__(self, evaluator, generator, data_path, test_meta_path, img_dir, config_path):
        self.evaluator = evaluator
        self.generator = generator
        self.data_path = data_path
        self.test_meta_path = test_meta_path
        self.img_dir = Path(img_dir)

        self.cfg = Config(config_path, default="lffont/cfgs/defaults.yaml")

        self.env = load_lmdb(self.data_path)
        self.env_get = lambda env, x, y, transform: transform(
            read_data_from_lmdb(env, f'{x}_{y}')['img']
        )

        self.test_meta = load_json(self.test_meta_path)
        self.dec_dict = load_json(self.cfg.dec_dict)


        ref_unis = self.test_meta["ref_unis"]
        gen_unis = self.test_meta["gen_unis"]
        gen_fonts = self.test_meta["gen_fonts"]
        self.target_dict = {f: gen_unis for f in gen_fonts}

        self.loader = get_fact_test_loader(
            self.env, self.env_get, self.target_dict, ref_unis, self.cfg, None, self.dec_dict,
            setup_transforms(self.cfg)[1], ret_targets=False, num_workers=self.cfg.n_workers, shuffle=False
        )[1]

    def run_evaluation(self):
        return self.evaluator.save_each_imgs(
            self.generator, self.loader, save_dir=self.img_dir, phase=self.cfg.phase, reduction='mean'
        )