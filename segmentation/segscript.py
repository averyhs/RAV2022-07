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
import numpy as np
import timeit
import datetime

datadirs = ['../../SampleData/VerySmallExample', '../../SampleData/S01_nodrugs', '../../SampleData/S02_withdrug']

# Structure of data directory:
#     SampleData
#     ├─── VerySmallExample
#     |    ├─── images
#     |    |   ├─── 000000000000.png
#     |    |   ├─── 000000000001.png
#     |    |   ├─── 000000000002.png
#     |    |   ├─── ...
#     |    |   └─── xxxxxxxxxxxx.png
#     |    ├─── annotations.json
#     |    └─── masks.npz             # may not be here if it has not yet been generated
#     ├─── S01
#     |    ├─── images
#     |    └─── ...
#     ├─── S02
#     └─── ...

images = []
masks = []

for d in datadirs:
    print(d)

    # Read the images (X)
    # -------------------
    print('    Reading images')
    d_images = list(io.ImageCollection(os.path.join(d, 'images/*'), conserve_memory=True))

    # Read the ground truth and 
    # Read or create masks (Y)
    # -------------------------
    d_masks_filepath = os.path.join(d, 'masks.npz')
    if os.path.exists(d_masks_filepath):
        print('    Reading masks')
        d_masks = np.load(d_masks_filepath)
        d_masks = [d_masks[x] for x in d_masks.files]
    else:
        print('    Creating masks')
        coco = Data.read_coco_file(os.path.join(d, 'annotations.json'))
        d_masks = Data.coco_to_masks(coco, d_masks_filepath)

    # Add to total dataset
    images.extend(d_images)
    masks.extend(d_masks)

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
t0 = timeit.default_timer()
model_out = model.train(data.dataloader, savefile='model.pt', recordfile='records.npz')
t1 = timeit.default_timer()

print('Training time: {time}'.format(time=datetime.timedelta(seconds=t1-t0)))
