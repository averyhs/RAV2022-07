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

datadir = '../../SampleData/VerySmallExample'

# Structure of data directory:
#     SampleData <all_data>
#     ├─── VerySmallExample <datadir>
#     |    ├─── images
#     |    |   ├─── 000000000000.png
#     |    |   ├─── 000000000001.png
#     |    |   ├─── 000000000002.png
#     |    |   ├─── ...
#     |    |   └─── xxxxxxxxxxxx.png
#     |    ├─── annotations.json
#     |    ├─── masks.npz             # may not be here if it has not yet been generated
#     |    └─── VerySmallExample.zip
#     ├─── <other_datadir>
#     └─── ...


# Structure of CellSium COCOOutput dir:
#     COCOOutput-simulate
#     ├─── annotations.json        # COCO formatted JSON file
#     └─── train                   # simulated microscope images
#          ├─── 000000000000.png
#          ├─── 000000000001.png
#          ├─── 000000000002.png
#          ├─── ...
#          └─── xxxxxxxxxxxx.png

# Read the images (X)
# -------------------
print('Reading images')
images = list(io.ImageCollection(os.path.join(datadir, 'images/*'), conserve_memory=True))

# Read the ground truth and 
# Read or create masks (Y)
# -------------------------
masks_filepath = os.path.join(datadir, 'masks.npz')
if os.path.exists(masks_filepath):
    print('Reading masks')
    masks = np.load(masks_filepath)
    masks = [masks[x] for x in masks.files]
else:
    print('Creating masks')
    coco = Data.read_coco_file(os.path.join(datadir, 'annotations.json'))
    masks = Data.coco_to_masks(coco, masks_filepath)

masks = [masks[x][:,:,1] for x in range(len(masks))] # single channel for now

# Data
# ----
print('Processing and loading data')
data = Data(images, masks)

# # see an example
# io.imsave('test-1_X.png', data.X[5])
# io.imsave('test-1_Y.png', data.Y[5][:,:,1])

# # Test that unpatch runs without errors
# data.unpatch()

# # Test patch and unpatch 
# # (not thorough, just quick check that images seem to be getting split and put back correctly)
# io.imsave('test-2_X.png', data.X[-1])
# io.imsave('test-2_Y.png', data.Y[-1][:,:,2])

# Model
# -----
print('Initializing a model')
model = Model() # create a Model instance

# Train
# ------
print('Commence Training!')
t0 = process_time()
model_out = model.train(data.dataloader)#, savefile='model.pt', recordfile='records.npz')
t1 = process_time()

print('Training time: {time}'.format(time=datetime.timedelta(seconds=t1-t0)))
