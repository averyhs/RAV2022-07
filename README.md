Mycobacterial colony simulation and segmentation from time-lapse microscopy
===========================================================================
### Project overview
This project is an initial exploration of image analysis methods for microscope images of microcolonies of rod-shaped mycobacteria, in 3 avenues. The objective is ultimately to identify and characterize each individual cell.

```
         --- Simulation ...
       /
o ----                          --- Image filtering ...
       \                      /
         --- Segmentation --- 
                              \
                                --- Machine learning ...
```

Simulation was explored to address the challenge of obtaning or generating ground truth data. We want ground truth data for testing how good a segmentation method is, and for training machine learning models. Can we artificially generate images to that resemble the actual microscope images closely enough to be useful?

Segmentation is the first step to identifying and characterizing individual cells. Image segmentation is the process of partitioning an image into regions of pixels that belong to different classes (semantic segmentation), or different instances of objects (instance segmentation). One method investigated was using an image filter that exploited the characteristics of the particular images dealt with in this project (cells are small and width doesn't vary much between instances, so maybe we can isolate the spatial frequency occupied by the cells). The other method was using a machine learning architecture called [U-Net-Id](https://doi.org/10.3390/rs12101544) (_F. H. Wagner, R. Dalagnol, Y. Tarabalka, T. Y. Segantine, R. Thomé, and M. C. Hirye, "U-Net-Id, an instance segmentation model for building extraction from satellite images — case study in the joanópolis city, brazil," Remote Sensing, vol. 12, no. 10, p. 1544, 2020._), which was developed for segmentaing satellite images of buildings - a different application with comparable image characteristics. 

This documentation describes the technical aspects of the projects and how to use it. It does not deal with associated theory and applications.

### Notes on scripts and environments
* Shell scripts and commands are written for Linux shells (Bash, Zsh, etc) so if you're using something different you may need to make some changes.

* All other scripts are Python3.

* Each folder (images, simulation, segmentation) contains a YAML file `environment.yml` that can be used to create the environment needed for each part of the project. The environments are separated to allow any part of the project to be used independently in other aplications, but it may be a little clumsy when running everything together.

* Using conda, each environment can be created by running `conda env create -f environment.yml` At the end of the setup, the output should include a message prompting you to activate the environment. The name of the environment can be seen in that message or at the top of the YAML file.

* Most of the code has undergone only just enough testing to see that it would meet my needs for the project. 

Setting up and working with images (🗀 images)
----------------------------------------------

### Selecting field of view
When selecting a smaller field of view than the original microscope images i used MicrobeJ to select and export the series of zoomed in frames, but other tools  could also be used. I generally looked for an area where there were some solitary cells and some dense microcolonies, but not so dense that cells were difficult to differentiate by eye.

### Alignment of images
There is drifting during microscope imaging over time, so there is some misalignment between frames. The alignment script I used (`image_alignment.py`) is a script written by Heather S. Deter and Dr. Nicholas C. Butzin for the CellTracking project, and can be found in [https://github.com/hdeter/CellTracking](https://github.com/hdeter/CellTracking). It works very well, but takes a folder of images, not a multi-frame TIFF. So I have written `tiffsep.py` and `tiffjoin.py` to convert between the two (folder of single-frame TIFF files <-> multi-frame TIFF file). 

If starting with a TIFF file containing all frames of the series, run the following commands _(commands for bash or zsh)_:

1. Separate the TIFF file into multiple images. (You will be prompted for the name of the input file. The output folder for the separation will be called `separated_tiff_images`. )
```
python tiffsep.py
```

2. Align the images frame-to-frame (I'll call the output folder `aligned`)
```
python image_alignment.py separated_tiff_images aligned
```

3. Join the images back into a single file. (You will be prompted for the name of the input folder. The output file will be called `joined_tiff_ims.py`. )
```
python tiffjoin.py
```

### Viewing timelapse as video
`makevideo.py` converts a series of frames, given as either a single TIFF file or a folder of images, to MP4 and GIF.

You will need to have [FFmpeg](https://ffmpeg.org/) installed. Note that this Python script runs shell commands. 

Run the following command (from within the `images` directory):
```
python makevideo.py [in] [out]
```

`[in]` can be a single multi-frame TIFF file or a folder of images named in chronological order. `[out]` is the folder to which the output files will be written. 

If the output folder does not exist it will not be created. Note that files with the same names as the output files (`cell_timelapse.mp4`, `cell_timelapse.gif`) will be overwritten.

### PNG to TIFF
The CellSium simulation script outputs PNG (CellSium can be made to output TIFF, but the color of the images changes, possibly due to datatype conversion). `pngseries2tiff.py` converts the CellSium output to a TIFF file.

Use
```
python pngseries2tiff.py [folder containing the PNG images]
```

Simulation
----------
The simulation is performed by [CellSium](https://cellsium.readthedocs.io/). From its documentation, "CellSium is a cell simulator developed for the primary application of generating realistically looking images of bacterial microcolonies, which may serve as ground truth for machine learning training processes." 

### Running the simulation



### Modifying the cell model and simulation parameters





~

~

~

~




















