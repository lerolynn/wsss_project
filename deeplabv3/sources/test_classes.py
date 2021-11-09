import cv2
import os
import numpy as np

#  Store the number of labels each image has
img_label_count = {}
with open('test_class.txt','r') as train_file:
    train_cls_labels = train_file.readlines()
    print(train_cls_labels)
    # for img_label in train_cls_labels:
    #     img_label_list = img_label.split(" ")
    #     img_label_count[img_label_list[0]] = len(img_label_list) -1
    #     print(img_label)

data_dir = "/home/lerolynn/smu/test/deeplabv3/results_dcrf/"
# img_dir = "/home/lerolynn/smu/test/deeplabv3/sample_dataset/train/Images"
# count = 0
for img_filename in train_cls_labels:
    png_filename = img_filename.split(".")[0] + ".png"
#     img_filepath =  os.path.join(data_dir, png_filename)

    print(png_filename)
    label_image = cv2.imread(os.path.join(data_dir,png_filename), cv2.IMREAD_GRAYSCALE)
    # raw_image = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
    # print(raw_image.shape)
    print(label_image.shape)
    # if raw_image.shape != label_image.shape:
    #     print(img_filename)
    print(set(list(label_image.flatten())))
#     if all(i >= 0 and i <= 103 for i in set(raw_image)):
#         continue
#     else:
#         print(png_filename)
#         print(set(raw_image))
#         count+=1

# print(count)
