import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image

import custom_model
import glob
import os
import cv2

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

# Number of classes in the dataset
num_classes = 104

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, input_size = custom_model.initialize_model(num_classes, keep_feature_extract=True, use_pretrained=False)

state_dict = torch.load("training_output/best_DeepLabV3_bestmodel.pth", map_location=device)

model = model.to(device)
model.load_state_dict(state_dict)
model.eval()

transforms_image =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

img_files = glob.glob(os.path.join("sample_dataset/test",'test2','*.*'))
# print(img_files)
for img_file in img_files:
    img_name = img_file.split("/")[-1].split(".")[0]
    image = cv2.imread(img_file)

    orig_dim = image.shape

    width = int(image.shape[1] * 0.3)
    height = int(image.shape[0] * 0.3)

    dim = (width, height)
    red_image = cv2.resize(image, (width, height)).squeeze()
    # red_image = red_image.transpose(2,0,1)

    image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    image = transforms_image(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = model(image)["out"]
    np_output = outputs.detach().cpu().numpy()
    
    print(np_output.shape)
    dcrf_output = outputs.detach().cpu().numpy().squeeze()
    # dcrf_output[dcrf_output < 0] = 0
    # dcrf_output = -np.log(dcrf_output)
    # print(dcrf_output)
    # print(red_image.shape)

    # d = dcrf.DenseCRF2D(red_image.shape[1], red_image.shape[0], 104)
    # print("outputs")
    # print(dcrf_output.shape)
    # U = utils.unary_from_softmax(dcrf_output)
    # d.setUnaryEnergy(U)

    # print("USHAPE")
    # print(U.shape)

    # d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
    #                       normalization=dcrf.NORMALIZE_SYMMETRIC)

    # d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=red_image,
    #                        compat=10,
    #                        kernel=dcrf.DIAG_KERNEL,
    #                        normalization=dcrf.NORMALIZE_SYMMETRIC)
   
    # Run five inference steps.
    # Q = d.inference(5)
   
    # Find out the most probable class for each pixel.
    # MAP = np.argmax(Q, axis=0)
    # print(MAP.shape)
    # print(set(MAP))
    # preds = MAP.reshape((height,width))

    #  ==== UNCOMMENT 
    _, preds = torch.max(outputs, 1)

    preds = preds.to("cpu")
    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)
    # ====================

    preds_np = cv2.resize(preds_np, (orig_dim[1],orig_dim[0]), interpolation=cv2.INTER_NEAREST)
    print(img_name)
    # print(preds_np.shape)
    cv2.imwrite(os.path.join("results_test_adam_bestmodel",img_name+".png"), np.uint8(preds_np))