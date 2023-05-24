import numpy as np
from skimage import io
import os

INDIR = input('Images folder: ')
OUTFILE = 'joined_tiff_ims.tiff'

# Input image files
imgs = list(os.path.join(INDIR,f) for f in os.listdir(INDIR) if f.endswith(('.tiff', '.tif'))) # list of image files in IMDIR
imgs.sort()

# Read images and add to list
imstack = []
for n in range(len(imgs)):
    img = io.imread(imgs[n])
    imstack.append(img)

imstack = np.array(imstack)

# np array to tiff
io.imsave(OUTFILE, imstack)
