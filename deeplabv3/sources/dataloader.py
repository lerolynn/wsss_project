# from __future__ import print_function
# from __future__ import division
import torch
import numpy as np
from torchvision import transforms
import os
import glob
from PIL import Image
import cv2

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(DataLoaderSegmentation, self).__init__()
        # print(folder_path)
        # self.img_names = []
        # self.label_files = []
        # self.img_files = []
        # with open('sorted_train_labels.txt','r') as train_file:
        #     train_cls_labels = train_file.readlines()
        #     for img_label in train_cls_labels:
        #         print(img_label)
        #         img_name = img_label.split(" ")[0].split(".")[0]
        #         jpg_name = img_name + ".jpg"
        #         png_name = img_name + ".png"
        #         self.img_names.append(img_name)
        #         self.img_files.append(os.path.join(folder_path, "Images", jpg_name))
        #         self.label_files.append(os.path.join(folder_path, "Labels", png_name))

        self.img_files = glob.glob(os.path.join(folder_path,'image','*.*'))
        self.label_files = []
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            label_filename_with_ext = f"{image_filename}.png"
            self.label_files.append(os.path.join(folder_path, 'label', label_filename_with_ext))
        
        print(self.img_files)

        # Data augmentation and normalization for training
        # Just normalization for validation
        if "val" == mode :
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
            ])
        else:
            self.transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Resize((224, 224)),
                    # transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1])
                ])

    def __getitem__(self, index):
            img_path = self.img_files[index]
            label_path = self.label_files[index]

            # print(img_path)
            # image = np.asarray(Image.open(img_path))
            # label = np.asarray(Image.open(label_path))

            image_np = cv2.imread(img_path)
            label_np = cv2.imread(label_path, cv2.COLOR_BGR2GRAY)

            # Concatenate image and label, to apply same transformation on both
            new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
            # print(new_shape)
            image_and_label_np = np.zeros(new_shape, image_np.dtype)
            # print(image_and_label_np.shape)
            image_and_label_np[:, :, 0:3] = image_np
            # print(image_and_label_np[:, :, 0:3].shape)
            image_and_label_np[:, :, 3] = label_np
            # print(image_and_label_np[:, :, 3].shape)

            # Convert to PIL
            image_and_label = Image.fromarray(image_and_label_np)

            # Apply Transforms
            image_and_label = self.transforms(image_and_label)

            # Extract image and label
            image = image_and_label[0:3, :, :]
            label = image_and_label[3, :, :].unsqueeze(0)

            # Normalize back from [0, 1] to [0, 255]
            label = label * 255
            #  Convert to int64 and remove second dimension
            label = label.long().squeeze()

            return image, label

    def __len__(self):
        return len(self.img_files)