# encoding: utf8
'''
Author: Peng Bo
Date: 2022-05-02 13:11:51
LastEditTime: 2022-05-29 20:52:11
Description: 

'''
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch
import cv2

img_size = 512

## get the position of landmarks
x_pos_idxs = np.array(range(img_size)).reshape(1, -1)
x_pos_idxs = np.repeat(x_pos_idxs, img_size, axis=0)
y_pos_idxs = np.array(range(img_size)).reshape(-1, 1)
y_pos_idxs = np.repeat(y_pos_idxs, img_size, axis=1)

def visualize_heatmap(imgs, gt_heatmaps, nrows=3, pred_heatmaps=None, size=(512, 512), save_path='test.jpg'):
    # pre-process the input imgs and masks
    fig, axes = plt.subplots(nrows=nrows, ncols=1)
  
    # original image
    img = torch.sum(imgs[0], axis=0).cpu().numpy() 
    img = np.reshape(img, newshape=size)
    img = np.transpose(img)
    img = np.stack([img,img,img], axis=-1)
    axes[0].imshow(img)

    # ground-truth mask
    gt_heatmap = torch.sum(gt_heatmaps[0], axis=0).cpu().numpy() 
    gt_heatmap = np.reshape(gt_heatmap, newshape=gt_heatmap.shape)
    gt_heatmap = np.transpose(gt_heatmap)
    axes[1].imshow(gt_heatmap)

    if pred_heatmaps is not None:
        # prediction heatmap
        pred_heatmap = torch.sum(pred_heatmaps[0], axis=0).cpu().numpy() 
        pred_heatmap = np.reshape(pred_heatmap, newshape=pred_heatmap.shape)
        pred_heatmap = np.transpose(pred_heatmap)
        axes[2].imshow(pred_heatmap)

    plt.savefig(save_path)


def visualize_heatmap(img, pred_heatmap=None, save_path='test.jpg'):
    img_size = pred_heatmap.shape[1]
    img = np.transpose(img.cpu().numpy())[:,:,0]
    img = 255*(img-np.min(img)) / (np.max(img) - np.min(img))
    img = img.astype(np.uint8)
    img = cv2.merge([img, img, img])
    pred_heatmap = np.transpose(pred_heatmap.cpu().numpy())
    # draw landmarks on image
    for i in range(pred_heatmap.shape[-1]):
        x_pos = int(np.sum(pred_heatmap[:,:,i] * x_pos_idxs) / np.sum(pred_heatmap[:,:,i]))
        y_pos = int(np.sum(pred_heatmap[:,:,i] * y_pos_idxs) / np.sum(pred_heatmap[:,:,i]))
        cv2.circle(img, (x_pos,y_pos), 2, (0,0,255), -1)

    cv2.imwrite(save_path, img)
