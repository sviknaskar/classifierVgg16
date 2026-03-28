import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def showImage(images, cols = 4):
    rows = len(images) // cols
    plt.figure(figsize=(14, 14* rows // cols))
    i = 1
    for img in images:
        plt.subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.astype(np.uint8))
        i+=1
    plt.show()


def applyMask(image, mask, color=np.array([0,255,255]), alpha = 0.5):
    for c in range(3):
        image[:,:,3] = np.where(mask == 1,
                                image[:,:,c] * (1-alpha) + alpha*color[c]*255,
                                image[:,:,c]
                                )
    return image



def swapCVImageChannels(image):
    if image.shape[-1] == 3:
        image = image[:,:,::-1]
    return image


imgDirPath = "D:/Projects/data/aquarium/AquariumCombined/train"
allImg = []
for i in os.listdir(imgDirPath):
    prefix = i.split(".")
    if prefix[-1] == 'jpg':
        allImg.append(imgDirPath+"/"+i)
    # print(prefix)
    # if not os.path.isdir(i):
    #     allImg.append(id)

img1 = cv2.imread(allImg[1])
img1 = swapCVImageChannels(img1)
# showImage(img1)
# img = img1[:,:,::-1]
plt.imshow(img1)
plt.show()