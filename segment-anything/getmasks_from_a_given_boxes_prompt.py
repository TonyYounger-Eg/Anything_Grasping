# coding:utf-8
#
# getmasks_from_a_given_boxes_prompt.py
#
#  Created on: 2023/9/4
#      Author: Tex Yan Liu
#
# description: 用框进行实例分割

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import sys
sys.path.append("../scripts")

sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        print("color", color)
    h, w = mask.shape[-2:]
    print("color.reshape(1, 1, -1)", color.reshape(1, 1, -1))
    print("mask.reshape(h, w, 1)", mask.reshape(h, w, 1))
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print("mask_image", mask_image)

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == '__main__':
    image = cv2.imread('jpeg/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    boxes = np.array([
        [1100, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ])
    predictor.set_image(image)
    input_boxes = torch.tensor(boxes, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()



