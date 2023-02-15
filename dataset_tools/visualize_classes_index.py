import os
import cv2 as cv
import numpy as np

path = "/root/data1/zbc/erfnet_pytorch/dataset/cityscapes/gtFine/train/aachen"
list = os.listdir(path)
print(len(list))
for item in list:
    if "labelTrainIds" not in item:
        continue
    file_path = os.path.join(path,item)
    print(file_path)
    image = cv.imread(file_path)
    image[image == 16] = 150
    image[image == 17] = 200
    image[image == 255] = 0

    cv.imwrite(os.path.join("/root/data1/zbc/erfnet_pytorch/dataset_tools/visualize_1617",item), image)

'''
image[image==3] =200
print(type(image))
print(image.shape)
cv.imwrite("./test.png",image)

'''