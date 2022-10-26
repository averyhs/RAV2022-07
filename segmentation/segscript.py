# ================================================================== #
# Segmentation script                                                #
#                                                                    #
# General script for running procedures and tests                    #
# ================================================================== #

# # Test making the 3 masks #
# # ----------------------- #

# import dataproc
# import matplotlib.pyplot as plt

# data = dataproc.read_coco_file('annotations.json')
# all_masks = dataproc.coco_to_masks(data, test=False)

# print(len(all_masks))
# print(all_masks[0].shape)

# sample = all_masks[-1]

# fig, axs = plt.subplots(1,3 , figsize=(12,4))
# axs[0].imshow(sample[:,:,0]), axs[0].set_title('cells')
# axs[1].imshow(sample[:,:,1]), axs[1].set_title('borders')
# axs[2].imshow(sample[:,:,2]), axs[2].set_title('inners')

# plt.show()



# Do stuff to run NN things #
# ------------------------- #

import os
import skimage.io as io
import matplotlib.pyplot as plt
import dataproc
from Model import Model
from time import process_time
import datetime

from params import  ROOT_DIR, \
                    patch_size, \
                    batch_size

datadir = 'Code/SampleData/VerySmallExample/COCOOutput-simulate'
# Structure or CellSium COCOOutput dir:
#     COCOOutput-simulate
#     ├─── annotations.json        # COCO formatted JSON file
#     └─── train                   # simulated microscope images
#          ├─── 000000000000.png
#          ├─── 000000000001.png
#          ├─── 000000000002.png
#          ├─── ...
#          └─── xxxxxxxxxxxx.png

# Make masks from COCO file
print('Creating masks')
coco = dataproc.read_coco_file(os.path.join(ROOT_DIR, datadir, 'annotations.json'))
masks = dataproc.coco_to_masks(coco, test=False)
masks3 = masks
masks = list(x[:,:,0] for x in masks) # For the basic original U-Net we only want the basic cell masks actually

# Read simulated microscope images into list via skimage ImageCollection (convert to list for consistency with masks)
print('Reading images')
images = list(io.ImageCollection(os.path.join(ROOT_DIR, datadir, 'train/*'), conserve_memory=True))

# Patchify images
print('Patchifying images')
images_patched = dataproc.patch_images(images, patch_size)
masks_patched = dataproc.patch_images(masks, patch_size)
# masks3_patched = dataproc.patch_images(masks3, patch_size, channels=3)

# Load data into DataLoader
print('Loading data')
data = dataproc.load_data(images_patched, masks_patched, batch_size)

# Create a Model instance
model = Model()

# Train!
print('Commence Training!')
t0 = process_time()
model_out = model.train(data, savefile='model.pt', recordfile='records.npz')
t1 = process_time()

print('Training time: {time}'.format(time=datetime.timedelta(seconds=t1-t0)))

