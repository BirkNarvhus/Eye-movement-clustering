import math
import sys

import cv2
import numpy as np
from PIL import Image
sys.path.append('C:\\Users\\birkn\\Documents\\bachlor\\eye-movement-classification')

from util.dataset_loader import OpenEDSLoader
from util.transformations import Crop_top, Crop, Normalize

transformations = [
    Crop_top(20),  # centers the image better
    Crop((256, 256)),
]
relative_path = "" #'../../'
root = relative_path + 'data/openEDS/openEDS'
save_path = relative_path + 'data/openEDS/openEDSSplit.npy'
loader = OpenEDSLoader(root, batch_size=8, shuffle=True, max_videos=None, save_path=save_path,
                       save_anyway=False,
                       transformations=transformations, sim_clr=False, split_frames=6)

train_loader, test_loader, _ = loader.get_loaders()
batch = train_loader.__next__()
batch = np.array(batch)
img = batch[0]
imgs = np.moveaxis(img, 0, -1)

for img in imgs:
    print(img.shape)
    img = np.array(img, dtype=np.uint8)

    inv = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(inv,kernel,iterations = 1)
    ret,thresh1 = cv2.threshold(erosion,220,255,cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) != 0:
        c = max(cnts, key = cv2.contourArea)
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img,center,radius,(255,0,0),2)

    cv2.circle(img, (int(x), int(y)), int(radius), (255, 255, 255), 20)
    cv2.imshow('Pupil Detector', img)
    cv2.waitKey(500)

c = cv2.waitKey()
cv2.destroyAllWindows()