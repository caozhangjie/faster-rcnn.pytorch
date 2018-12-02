
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg

import numpy as np
import random
import time
from scipy.misc import imread
import os
import os.path as opath

class JAADLoader(data.Dataset):
  def __init__(self, img_root, bbox_root, phase='train'):
    dir_list = os.listdir(img_root)
    self.phase=phase
    self.imgs = []
    for dir_ in dir_list:
        dir_id = int(dir_.split("_")[1].split(".")[0])
        if (phase == 'test' and dir_id > 250) or (phase == 'train' and dir_id <= 250):
            bbox = np.load(opath.join(bbox_root, "video_{:04d}.npy".format(dir_id))).item()
            for img_id in os.listdir(opath.join(img_root, dir_)):
                img_path = opath.join(img_root, dir_, img_id)
                idx = int(img_id.split(".")[0])
                poss = [ped['pos'] for ped in bbox["objLists"][idx-1] if ped['occl'][0] < 0.5]
                for pos in poss:
                    pos[0] = int(pos[0]+0.5)
                    pos[1] = int(pos[1]+0.5)
                    pos[2] = int(pos[2]+0.5)
                    pos[3] = int(pos[3]+0.5)
                    pos[2] = pos[0] + pos[2] - 1
                    pos[3] = pos[1] + pos[3] - 1              
                self.imgs.append([img_path, np.array(poss)])


  def __getitem__(self, index):
    img_path, pos = self.imgs[index]
    img = imread(img_path)
    im_info = torch.from_numpy(np.array([img.shape[0], img.shape[1], 1.0]))
    num_boxes = len(pos)
    img = img[:,:,::-1] - cfg.PIXEL_MEANS
    img = torch.from_numpy(img).float().permute(2,0,1).contiguous()
    if pos.shape[0] == 0:
        pos_l = np.zeros([1, 5])
    else:
        pos_l = np.zeros([pos.shape[0], 5])
        pos_l[:, 0:4] = pos
        pos_l[:, 4] = 1.0
    pos = torch.from_numpy(pos_l).float()
    if self.phase == 'train':
        return img, im_info, pos, num_boxes
    else:
        return img, im_info, pos, num_boxes, img_path

  def __len__(self):
    return len(self.imgs)
