# ------------------------------------------------------------------------------
# Batch tongue extraction: segment all images, output tongue-only (bg=black).
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import _init_paths
import models

# --- config ---
MODEL_PATH = 'output/tongue/pidnet_small_tongue/best.pt'
IMAGE_DIR = 'C:/python/tough/data/tongue'
OUTPUT_DIR = 'C:/python/tough/data/tongue_only'
INFER_SIZE = (1024, 1024)  # resize for stable inference, then map back
MODEL_NAME = 'pidnet_s'
NUM_CLASSES = 2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess(image, target_size):
    """Resize, normalize, convert to tensor."""
    h, w = image.shape[:2]
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)[:, :, ::-1]  # BGR -> RGB
    image = image / 255.0
    image -= mean
    image /= std
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0), (h, w)


def main():
    # Load model
    model = models.pidnet.get_pred_model(MODEL_NAME, NUM_CLASSES)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    # Strip 'model.' prefix (FullModel wrapper) and 'module.' (DataParallel)
    cleaned = {}
    for k, v in ckpt.items():
        new_k = k
        for prefix in ['model.', 'module.']:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    model.load_state_dict(cleaned, strict=False)
    model.cuda().eval()

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f'Found {len(image_files)} images. Output -> {OUTPUT_DIR}')

    with torch.no_grad():
        for fname in tqdm(image_files):
            img_path = os.path.join(IMAGE_DIR, fname)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f'  SKIP (unreadable): {fname}')
                continue
            orig_h, orig_w = img.shape[:2]

            tensor, (oh, ow) = preprocess(img, INFER_SIZE)
            tensor = tensor.cuda()

            pred = model(tensor)
            # pred shape: (1, 2, H, W) — resize mask back to original
            pred = F.interpolate(pred, size=(oh, ow),
                                 mode='bilinear', align_corners=True)
            mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Apply mask: keep tongue pixels, set background to black
            tongue_only = img.copy()
            tongue_only[mask == 0] = 0

            out_path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(out_path, tongue_only)

    print(f'Done. {len(image_files)} images saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
