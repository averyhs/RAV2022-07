# ================================================================== #
# Segmentation script                                                #
#                                                                    #
# General script for running procedures and tests                    #
# ================================================================== #

import os
import skimage.io as io
import matplotlib.pyplot as plt
import dataproc
from Model import Model
from time import process_time
import numpy as np

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
mask_filepath = os.path.join(ROOT_DIR, datadir, 'masks.npz')
masks = dataproc.coco_to_masks(coco, mask_filepath)

masks_read = np.load(mask_filepath)
print(masks_read.files)
plt.imshow(masks_read['arr_0'][:,:,1])
plt.show()

masks3 = masks
masks = list(x[:,:,0] for x in masks) # For the basic original U-Net we only want the basic cell masks actually

# # Read simulated microscope images into list via skimage ImageCollection (convert to list for consistency with masks)
# print('Reading images')
# images = list(io.ImageCollection(os.path.join(ROOT_DIR, datadir, 'train/*'), conserve_memory=True))

# # Patchify images
# print('Patchifying images')
# images_patched = dataproc.patch_images(images, patch_size)
# masks_patched = dataproc.patch_images(masks, patch_size)
# # masks3_patched = dataproc.patch_images(masks3, patch_size, channels=3)

# # Load data into DataLoader
# print('Loading data')
# data = dataproc.load_data(images_patched, masks_patched, batch_size)

# # Create a Model instance
# model = Model()

# # Train!
# print('Commence Training!')
# t0 = process_time()
# model_out = model.train(data, savefile='model.pt', recordfile='records.npz')
# t1 = process_time()

# print('Training time: {time}'.format(time=datetime.timedelta(seconds=t1-t0)))

