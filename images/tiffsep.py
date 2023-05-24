import numpy as np
from skimage import io
from pathlib import Path

INFILE = input('Multi-image tiff file: ')
OUTDIR = 'separated_tiff_images'

# Read multi-image tiff
tiffs_img = io.imread(INFILE)

# Make output dir if it doesn't exist
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# Write individual images
ni = np.shape(tiffs_img)[0] # number of images in the tiff file
for n in range(ni):
    io.imsave(OUTDIR+f'/image_{n:03}.tiff', tiffs_img[n])
