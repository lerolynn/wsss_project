import cv2
import os
import numpy as np

#  Store the number of labels each image has
img_label_count = {}
with open('data/sorted_train_labels.txt','r') as train_file:
    train_cls_labels = train_file.readlines()

    for img_label in train_cls_labels:
        img_label_list = img_label.split(" ")
        img_label_count[img_label_list[0]] = len(img_label_list) -1

data_dir = "/home/lerolynn/smu/dl/dl_project/pseudo_mask/mask"
count = 0
for img_filename in img_label_count:
    png_filename = img_filename.split(".")[0] + ".png"
    img_filepath =  os.path.join(data_dir, png_filename)
    # print(png_filename)
    raw_image = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE).flatten().tolist()
    print(set(raw_image))
    if all(i >= 0 and i <= 103 for i in set(raw_image)):
        continue
    else:
        print(png_filename)
        print(set(raw_image))
        count+=1

print(count)
