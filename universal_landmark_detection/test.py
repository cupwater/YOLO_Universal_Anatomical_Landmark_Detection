# import numpy as np
# import pandas as pd
#
#
# test = np.load('C:/Users/jason/Desktop/runs/GU2Net/results/test_epoch067/chest/CHNCXR_0032_0_gt.npy')
# print(test)



from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('/home/ubuntu/liguanghui/YOLO_Universal_Anatomical_Landmark_Detection-main/universal_landmark_detection/CHNCXR_0001_0.png')
buf = np.array(im)
plt.hist(buf.flatten(), 256, normed = True)
plt.show()

