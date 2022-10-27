# ================================================================== #
# Segmentation script                                                #
#                                                                    #
# General script for running procedures and tests                    #
# ================================================================== #

import os
import skimage.io as io
import matplotlib.pyplot as plt
from Model import Model
from Data import Data
from time import process_time
import numpy as np
import datetime

from params import  ROOT_DIR

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

# Read the images <- X
# --------------------
print('Reading images')
images = list(io.ImageCollection(os.path.join(ROOT_DIR, datadir, 'train/*'), conserve_memory=True))

# Read the ground truth and Make the masks <- Y
# ---------------------------------------------
print('Creating masks')
mask_filepath = os.path.join(ROOT_DIR, datadir, 'masks.npz')

# # No masks exist yet:
# coco = dataproc.read_coco_file(os.path.join(ROOT_DIR, datadir, 'annotations.json'))
# masks = dataproc.coco_to_masks(coco, mask_filepath)

# Masks exist:
masks = np.load(mask_filepath)
masks = [masks[x] for x in masks.files]

# masks = list(x[:,:,0] for x in masks) # For the basic original U-Net we only want the basic cell masks actually

# Data
# ----
print('Processing and loading data')
data = Data(images, masks)

data.unpatch()

plt.imshow(data.X[0])
plt.show()

plt.imshow(data.Y[0])
plt.show()