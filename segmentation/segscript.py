# ================================================================== #
# Segmentation script                                                #
#                                                                    #
# General script for running procedures and tests                    #
# ================================================================== #

import dataproc
import matplotlib.pyplot as plt

data = dataproc.read_coco_file('annotations.json')
masks = dataproc.coco_to_masks(data, test=True)

fig, axs = plt.subplots(1,3 , figsize=(12,4))
axs[0].imshow(masks[:,:,0]), axs[0].set_title('cells')
axs[1].imshow(masks[:,:,1]), axs[1].set_title('borders')
axs[2].imshow(masks[:,:,2]), axs[2].set_title('inners')

plt.show()