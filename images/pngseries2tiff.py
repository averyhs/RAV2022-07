import numpy as np
from skimage import io
import os, sys

def main(IN):
    OUT = 'png2tiff_images.tiff'

    # Input image files
    pngs = list(os.path.join(IN,f) for f in os.listdir(IN) if f.endswith('.png')) # list of pngs (as paths) in image folder
    pngs.sort()

    # Get required shape of np array
    sample_img = io.imread(pngs[0])
    imshape = np.shape(sample_img)
    imdtype = sample_img.dtype

    # Read images and add to list
    imstack = []
    for n in range(len(pngs)):
        img = io.imread(pngs[n])
        imstack.append(img)

    imstack = np.array(imstack)
    # imstack = imstack.astype(imdtype)

    # np array to tiff
    io.imsave(OUT, imstack)

if __name__ == '__main__':
    IN = sys.argv[1] # Folder containing the PNG images

    main(IN)
