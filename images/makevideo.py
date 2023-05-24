import numpy as np
from skimage import io
import imageio
import os, sys

def main(IN, OUT):
    '''
    Make video from images to visualize timelapse.

    Args:
        IN (str): Path to input images. Can be folder of images (frame order based on file names) or a TIFF file of all the images
        OUT (str): Folder for output
    '''

    # READ FILE(S) AND GET LIST OF FRAMES
    frames = []

    # If input is a FILE (assume it is a multi-frame TIFF file):
    if os.path.isfile(IN):
        tiffs_img = io.imread(IN) # Read the file
        ni = np.shape(tiffs_img)[0] # Get num of images in the tiff file
        
        # List of frames for video
        for n in range(ni):
            frames.append(tiffs_img[n])
    
    # If input is a FOLDER (assume it is a directory of single inmages, named in order):
    if os.path.isdir(IN):
        imgs = list(os.path.join(IN,f) for f in os.listdir(IN))
        imgs.sort()

        # Read images and add to list of frames for video
        for n in range(len(imgs)):
            frames.append(io.imread(imgs[n]))

    # WRITE IMAGES TO TEMPORARY FOLDER - NUMBERED PNGS
    os.system(f'mkdir {TMP}')
    for n in range(len(frames)):
        io.imsave(TMP+f'/{n:04}.png', frames[n])

    # CREATE VIDEOS USING FFMPEG
    os.system(f'mkdir {OUT}')
    os.system('ffmpeg -r 8 -i {} -f mp4 -vcodec libx264 -y {}'.format(TMP+'/%04d.png', OUT+'/cell_timelapse.mp4')) # create MP4 video
    os.system('ffmpeg -r 8 -i {} -f gif -y {}'.format(TMP+'/%04d.png', OUT+'/cell_timelapse.gif')) # create GIF image

    os.system(f'rm -r {TMP}') # remove temporary folder of PNG files

if __name__ == "__main__":
    IN = sys.argv[1]
    OUT = sys.argv[2]
    TMP = 'tmp_imgs'

    main(IN, OUT)
