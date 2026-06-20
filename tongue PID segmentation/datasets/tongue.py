# ------------------------------------------------------------------------------
# Tongue dataset for binary semantic segmentation (background vs tongue).
# Adapted from CamVid dataset implementation.
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset


class Tongue(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=2,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=1280,
                 crop_size=(640, 640),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Tongue, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.ignore_label = ignore_label

        # No class weights for binary segmentation (balanced by OHEM)
        self.class_weights = None

        self.bd_dilate_size = bd_dilate_size

        # Build file list from the paired samples in tongue/ and mask/ dirs
        self.files = self.read_files()

    def read_files(self):
        """Build file list from paired (image, mask) samples.

        Only uses the 996 samples that have both an image in data/tongue/
        and a mask in data/mask/ (matching by filename).
        Splits into train/val based on list_path parameter.
        """
        tongue_dir = os.path.join(self.root, 'tongue')
        mask_dir = os.path.join(self.root, 'mask')

        # Find all paired samples (matching filenames)
        all_samples = []
        for fname in sorted(os.listdir(tongue_dir)):
            if not fname.lower().endswith('.jpg'):
                continue
            base = os.path.splitext(fname)[0]
            mask_fname = base + '.png'
            mask_path = os.path.join(mask_dir, mask_fname)
            if os.path.exists(mask_path):
                all_samples.append(base)

        # Deterministic 80/20 split
        np.random.seed(42)
        indices = np.random.permutation(len(all_samples))
        split = int(len(all_samples) * 0.8)

        if 'train' in self.list_path:
            selected = [all_samples[i] for i in indices[:split]]
        else:
            selected = [all_samples[i] for i in indices[split:]]

        files = []
        for base in sorted(selected):
            files.append({
                "img": os.path.join(tongue_dir, base + '.jpg'),
                "label": os.path.join(mask_dir, base + '.png'),
                "name": base
            })

        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = Image.open(item["img"]).convert('RGB')
        image = np.array(image)
        size = image.shape

        label = Image.open(item["label"]).convert('L')
        label = np.array(label, dtype=np.uint8)
        # Convert mask: 0 stays 0 (background), 255 -> 1 (tongue)
        label[label == 255] = 1

        # For validation (no multi_scale), resize to crop_size to ensure
        # consistent dimensions and fit within VRAM
        if not self.multi_scale:
            import cv2
            image = cv2.resize(image, (self.crop_size[1], self.crop_size[0]),
                               interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.crop_size[1], self.crop_size[0]),
                               interpolation=cv2.INTER_NEAREST)

        image, label, edge = self.gen_sample(image, label,
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            # Save as binary mask: 0=background, 255=tongue
            save_img = Image.fromarray(preds[i] * 255)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
