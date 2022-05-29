'''
Author: Peng Bo
Date: 2022-04-27 16:09:11
LastEditTime: 2022-05-22 18:36:23
Description: 

'''
import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from ..utils import gaussianHeatmap, transformer


class ChestInmemory26LMS(data.Dataset):

    def __init__(self, prefix, phase, transform_params=dict(), sigma=5, num_landmark=26, size=[512, 512], use_abnormal=True, chest_set=None, exclude_list=None, use_background_channel=False):

        self.transform = transformer(transform_params)
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel
        self.prefix = prefix

        self.imglist = open(os.path.join(prefix, 'imglist.txt')).readlines()
        self.imglist = [line.strip() for line in self.imglist]

        n = len(self.imglist)
        train_num = 270  # round(n*0.7) # 180
        val_num = 76  # round(n*0.1)  # 24
        test_num = n - train_num - val_num
        if phase == 'train':
            train_num = 306  # round(n*0.7) # 180
            self.indexes = self.imglist[:train_num]
        elif phase == 'validate':
            train_num = 270  # round(n*0.7) # 180
            val_num = 76  # round(n*0.1)  # 24
            test_num = n - train_num - val_num
            self.indexes = self.imglist[train_num:-test_num]
        elif phase == 'test':
            train_num = 270  # round(n*0.7) # 180
            val_num = 76  # round(n*0.1)  # 24
            test_num = n - train_num - val_num
            # self.indexes = files[-test_num:]
            self.indexes = self.imglist
        else:
            raise Exception("Unknown phase: {phase}".fomrat(phase=phase))
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(size))
        self.readLandmark = np.loadtxt(os.path.join(prefix, 'landmarks.txt')).reshape(-1, num_landmark*2)
        self.__readAllData__()


    def __readAllData__(self):
        self.all_images  = []
        self.all_imgsize = []
        for i in range(self.__len__()):
            name = self.indexes[i]
            ret = {'name': name}
            img, origin_size = self.readImage(
                os.path.join(self.prefix, name))
            self.all_images.append(img)
            self.all_imgsize.append(origin_size)
        return
        
    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}
        img, origin_size = self.all_images[index], self.all_imgsize[index]
        points = self.readLandmark[index]
        points = [tuple([int(points[2*i]*origin_size[0]), int(points[2*i+1]*origin_size[1])]) for i in range(int(points.shape[0]/2))]
        li = [self.genHeatmap(point, self.size) for point in points]
        if self.use_background_channel:
            sm = sum(li)
            sm[sm > 1] = 1
            li.append(1-sm)
        gt = np.array(li)
        img, gt = self.transform(img, gt)
        ret['input'] = torch.FloatTensor(img)
        ret['gt'] = torch.FloatTensor(gt)
        return ret

    def __len__(self):
        return len(self.indexes)

    def readImage(self, path):
        '''Read image from path and return a numpy.ndarray in shape of cxwxh
        '''
        img = Image.open(path)
        origin_size = img.size

        # resize, width x height,  channel=1
        img = img.resize(self.size)
        arr = np.array(img)
        # channel x width x height: 1 x width x height
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.expand_dims(np.transpose(arr, (1, 0)), 0).astype(np.float)
        # conveting to float is important, otherwise big bug occurs
        for i in range(arr.shape[0]):
            arr[i] = (arr[i]-arr[i].mean())/(arr[i].std()+1e-20)
        return arr, origin_size

