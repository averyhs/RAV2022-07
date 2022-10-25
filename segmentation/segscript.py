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

# Set paths
ROOT_DIR = '/home/avery/Documents/2022/EEE4022_1' # absolute path of project directory
datadir = 'Code/SampleData/COCOOutput-simulate'

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
masks = list(x[:,:,0] for x in masks) # For the basic original U-Net we only want the basic cell masks actually

# Read simulated microscope images into list via skimage ImageCollection (convert to list for consistency with masks)
print('Reading images')
images = list(io.ImageCollection(os.path.join(ROOT_DIR, datadir, 'train/*'), conserve_memory=True))

# Test that images and masks have been read in correctly
print('Testing that images and masks have been read in correctly')
print('    Num images:',len(images))
plt.imshow(images[-1])
plt.show()
print('    Num masks:',len(masks))
plt.imshow(masks[-1])
plt.show()
