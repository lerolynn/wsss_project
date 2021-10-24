import os


path = '../mask_for150epochs'
files = os.listdir(path)
from PIL import Image

# print(files)
for f in files:
    print(f.split(".")[0])
    # print(os.path.join(path, f))
    # print(os.path.join(path, f.split(".")[0].split("_")[0]+'.png'))
    # os.rename(os.path.join(path, f),  os.path.join(path, f.split(".")[0].split("_")[0] + '.jpg'))

    im1 = Image.open(os.path.join(path, f))
    save_filename = '../png_masks/{}.png'.format(f.split(".")[0])
    im1.save(save_filename)
# for index, file in enumerate(files):
#     print(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))