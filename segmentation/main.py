from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import datahandler
from model import createDeepLabv3
from trainer import train_model


@click.command()
@click.option("--data-directory",
              default="data_dir1",
              help="Specify the data directory.")
@click.option("--exp_directory",
              default="food_dataset_out",
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")

def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))



def segOutput(model):
	import glob
	import os
	#/home/team9/public/img_dir/train
	for img_path in glob.glob(os.path.join("/home/team9/public/img_dir/test1", "*.jpg")):
		ino = 2
		# Read  a sample image and mask from the data-set
		#img = cv2.imread(img_path).transpose(2,0,1).reshape(1,3,320,480)
		img = cv2.imread(img_path)
		mask_path=img_path.replace(".jpg","_label.jpg")
		mask = cv2.imread(mask_path)
		with torch.no_grad():
    			a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)

		#numpyArray=a['out'].cpu().detach().numpy()
		# Plot the input image, ground truth and the predicted output
		plt.figure(figsize=(10,10))
		plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.2)
		numpyArray=a['out'].cpu().detach().numpy()[0][0]>0.2

		output_path=img_path.replace("data_dir/Images","food_dataset_out")
		output_path=img_path.replace(".jpg",".png")

		im = Image.fromarray(numpyArray)
		gray_img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		cv2.imwrite(output_path,gray_img)
		#im.save(output_path, "png")
		plt.title('Segmentation Output')
		plt.axis('off');
		plt.savefig('food_dataset_out/SegmentationOutput.png',bbox_inches='tight')




def main(data_directory, exp_directory, epochs, batch_size):
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3(103)
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights_7_15.pt')
    #segOutput(model)


if __name__ == "__main__":
    main("data_dir1","food_dataset_out",15,4)
    import gc

    gc.collect()

    torch.cuda.empty_cache()
    
