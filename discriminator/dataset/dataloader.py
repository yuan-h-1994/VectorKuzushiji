from torch.utils.data import Dataset
import os
import json
import random
import numpy as np
import skimage.transform as transform
from skimage.io import imread
import skimage.color as color
import torchvision.transforms.functional as F
import torch


class WordImageLoader(Dataset):
    def __init__(self, dataset_path, image_set='train'):
        self.root = dataset_path
        self.eval = 0 if image_set == "train" else 1
        if self.eval:
            with open(os.path.join(self.root, "val_anno.json"), "r") as ann_json:
                self.ann = json.loads(ann_json.read())
        else:
            with open(os.path.join(self.root, "train_anno.json"), "r") as ann_json:
                self.ann = json.loads(ann_json.read())
        self.idx = list(self.ann.keys())
        if image_set == 'train':
            random.shuffle(self.idx)

    def __getitem__(self, index):
        img_name = self.idx[index]
        class_label = self.ann[self.idx[index]]

        img = self.rgb_img_read(os.path.join(self.root, img_name))

        target = torch.tensor(class_label)
        # return_dict["img_path"] = img_path[0]

        img_normal = F.normalize(F.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # return img_normal, target
        return [img_normal, img], target

    def _crop(self, img_path, bbox, context_expansion, img_side=224):
        img = self.rgb_img_read(img_path)
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_center = (x1+x2)/2
        y_center = (y1+y2)/2

        widescreen = True if w > h else False

        if not widescreen:
            img = img.transpose((1, 0, 2))
            x_center, y_center, w, h = y_center, x_center, h, w

        x_min = int(np.floor(x_center - w * (1 + context_expansion) / 2.))
        x_max = int(np.ceil(x_center + w * (1 + context_expansion) / 2.))

        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(img_side) / patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1, preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)

        starting_point = [x_min, y_min - top_margin]

        if not widescreen:
            new_img = new_img.transpose((1, 0, 2))
            starting_point = [y_min - top_margin, x_min]
        
        return_dict = {
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen
        }

        return new_img, return_dict

    def rgb_img_read(self, img_path):
        """
        Read image and always return it as a RGB image (3D vector with 3 channels).
        """
        img = imread(img_path)
        if len(img.shape) == 2:
            img = color.gray2rgb(img)

        # Deal with RGBA
        img = img[..., :3]

        if img.dtype == 'uint8':
            # [0,1] image
            img = img.astype(np.float32) / 255
        return img

    def __len__(self):
        return len(self.idx)


def build_dataset(args, image_set):
    root = args.dataset_path
    dataset = WordImageLoader(root, image_set)
    return dataset
