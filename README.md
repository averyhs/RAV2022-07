Mycobacterial colony simulation and segmentation from time-lapse microscopy
===========================================================================
This project is an initial exploration of image analysis methods for microscope images of microcolonies of rod-shaped mycobacteria, in 3 avenues. 

```
         --- Simulation ...
       /
o ----                          --- Image filtering ...
       \                      /
         --- Segmentation --- 
                              \
                                --- Machine learning ...
```

Ground truth data is a challenge (we want it for testing how good a method is, and for training machine learning models). Simulation is an option for obtaining data for training and testing. How closely can we get the artificially generated images to resemble the actual microscope images?

The objective is to identify and characterize each 

Setting up and preprocessing the data
-------------------------------------
This is the process i followed to prepare the data i used in the project.

I started with TIFF files taken directly from the microscope, showing different fields of view of the platform the microcolonies were grown on. Each TIFF file contained 80 frames of a timelapse. I aligned the frames, because there was drifting during the microscope imaging so there was some misalginment between frames. I then zoomed in and extracted a portion of the image where the microcolonies became dense, but still had fairly good contrast and definition (it's very difficult to segment when the microcolonies get super large, and those cases were not of the most interest for analysis anyway).


### _Sidenote - other file type to TIFF_

_Early on i was using JPG images. I wrote a Python script to convert a series of JPG images to a single TIFF file. If you want to follow my process using a different image file format it may be of interest. I will include the JPG 2 TIFF script and also a modified version that works for PNGs._

_Specify the name folder where the images are in the `IMDIR` variable in line 7. Note that the images will be sorted by filename._

_For JPG images:_
```
import numpy as np
from skimage import io
from skimage import exposure
import os

# Specify directory
IMDIR = 'folder-name' # path to image directory, relative to ROOTDIR

# Input image files
jpgs = list(os.path.join(IMDIR,f) for f in os.listdir(IMDIR) if f.endswith('.jpg')) # list of jpgs (as paths) in IMDIR
jpgs.sort()

# Get required shape of np array
sample_img = io.imread(jpgs[0])
imshape = np.shape(sample_img)

# Read jpg images and add to numpy stack (specifying datatype)
imstack = np.empty([len(jpgs), imshape[0], imshape[1]], dtype=sample_img.dtype)
for n in range(len(jpgs)):
    img = io.imread(jpgs[n])
    imstack[n,:,:] = img

# np array to tiff
io.imsave('images.tiff', imstack)
```

_For PNG images:_
```
import numpy as np
from skimage import io
from skimage import exposure
import os

# Specify directory
IMDIR = 'folder-name' # path to image directory, relative to ROOTDIR

# Input image files
pngs = list(os.path.join(IMDIR,f) for f in os.listdir(IMDIR) if f.endswith('.png')) # list of pngs (as paths) in IMDIR
pngs.sort()

# Get required shape of np array
sample_img = io.imread(pngs[0])
imshape = np.shape(sample_img)

# Read png images and add to numpy stack (specifying datatype)
imstack = np.empty([len(pngs), imshape[0], imshape[1], imshape[2]], dtype=sample_img.dtype)
for n in range(len(pngs)):
    img = io.imread(pngs[n])
    imstack[n,:,:,:] = img

# np array to tiff
io.imsave('images.tiff', imstack)
```

Simulation
----------
The simulation is performed by CellSium





























