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

from params import  ROOT_DIR, \
                    patch_size

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

# # Test that images and masks have been read in correctly
# print('Testing that images and masks have been read in correctly')
# print('    Num images:',len(images))
# plt.imshow(images[-1])
# plt.show()
# print('    Num masks:',len(masks))
# plt.imshow(masks[-1])
# plt.show()

# Patchify images
print('Patchifying images')
images_patched = dataproc.patch_images(images, patch_size)
masks_patched = dataproc.patch_images(masks, patch_size)

# Test - see a sample patch result for images and for masks
print('Check patch results')

fig,axs = plt.subplots(3,3)
axs[0,0].imshow(images_patched[0], cmap='gray')
axs[0,1].imshow(images_patched[1], cmap='gray')
axs[0,2].imshow(images_patched[2], cmap='gray')
axs[1,0].imshow(images_patched[3], cmap='gray')
axs[1,1].imshow(images_patched[4], cmap='gray')
axs[1,2].imshow(images_patched[5], cmap='gray')
axs[2,0].imshow(images_patched[6], cmap='gray')
axs[2,1].imshow(images_patched[7], cmap='gray')
axs[2,2].imshow(images_patched[8], cmap='gray')
fig.suptitle('Image patches')
plt.show()

fig,axs = plt.subplots(3,3)
axs[0,0].imshow(masks_patched[0], cmap='gray')
axs[0,1].imshow(masks_patched[1], cmap='gray')
axs[0,2].imshow(masks_patched[2], cmap='gray')
axs[1,0].imshow(masks_patched[3], cmap='gray')
axs[1,1].imshow(masks_patched[4], cmap='gray')
axs[1,2].imshow(masks_patched[5], cmap='gray')
axs[2,0].imshow(masks_patched[6], cmap='gray')
axs[2,1].imshow(masks_patched[7], cmap='gray')
axs[2,2].imshow(masks_patched[8], cmap='gray')
fig.suptitle('Mask patches')
plt.show()