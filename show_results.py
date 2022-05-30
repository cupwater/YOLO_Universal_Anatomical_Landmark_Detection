'''
Author: Peng Bo
Date: 2022-05-27 05:08:30
LastEditTime: 2022-05-28 22:25:29
Description: 

'''
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pdb

# imglist = open('imglist_6landmarks.txt').readlines()
# prefix = 'data/chest'

semantic_list = open('data/26landmarks_semantic.txt').readlines()
semantic_list = [l.split(' ')[0] for l in semantic_list]

imglist = open('imglist.txt').readlines()
prefix = 'data/chest_26landmarks'

for imgpath in imglist:
    _path = imgpath.strip().replace('png', 'jpg')
    img = cv2.imread(f'data/{_path}')
    # pdb.set_trace()
    img = cv2.resize(img, (512, 512))
    # pdb.set_trace()
    landmarks_path = imgpath.replace('png', 'txt')
    landmarks_path = f'{prefix}/labels/{landmarks_path.strip()}'
    with open(landmarks_path) as fin:
        lms = fin.readlines()[1:]
        lms = [l.strip() for l in lms]
        w, h = img.shape[0], img.shape[1]
        for idx, l in enumerate(lms):
            x, y = float(l.split(' ')[0]), float(l.split(' ')[1])
            x, y = int(w*x), int(h*y)
            cv2.circle(img, (x,y), 2, (255,0,0), 1)
            cv2.putText(img, semantic_list[idx], (x,y), 2, 0.5, (255,0,0), 1)
    
    cv2.imshow('img', img)

    key = cv2.waitKey(-1)
    if key != 27:
        continue