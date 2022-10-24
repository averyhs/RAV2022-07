# ================================================================== #
# Data processing                                                    #
#                                                                    #
# This file contains functions for preprocessing and postprocessing  #
# data for U-Net-Id.                                                 #
# ================================================================== #

import json
import numpy as np
import imantics
import cv2

def draw_masks(im_data):
    '''
    Create object mask, border mask, inner mask for an image (Intended for internal use).

    For a single image from a COCO-formatted dict, use the object masks to create a 
    border mask and an inner mask. The masks are returned as a single 3-channel image --TODO
    (as a numpy array), with each mask drawn in a separate channel.
    
    The border mask for each object is obtained by finding the edges of the object mask 
    and drawing them thicker, extending both inside and outside the object. The inner 
    mask for each object is obtained by performing an erosion operation on the object 
    mask.

    It is assumed that the image is square (equal height and width dims). Masks will 
    not be correctly produced if image is not square.

    Args:
        im_data (imantics image): The mask image data from which to produce masks
    
    Returns:
        ndarray: Masks as 3-channel image (numpy array)
    '''
    # Borders: for borders, images will be expanded and then cropped
    # so that outlines will not be drawn along the edges (for objects cut off by image edge)
    # Set params:
    thickness = 3 # thickness of border to draw
    pad = thickness
    
    # Inners: inners are obtained by erosion of cell masks
    # The value of gap is the side length of a square erosion kernel
    # It should be just enough to separate touching objects - so 3 should be good almost always
    # even values result in the eroded mask being offset from center (not sure why)
    gap = 3
    
    # Cell and border masks initialized with zeros, padded as specified above
    # (these will be updated in the loop)
    cell_mask = np.zeros((im_data.height, im_data.width, 3)).astype(np.uint8) # RGB format for Imantics
    cell_mask.setflags(write=True)
    border_mask = np.zeros((pad+im_data.height+pad, pad+im_data.width+pad)).astype(np.uint8) # padded
    border_mask.setflags(write=True)
    inner_mask = np.zeros(border_mask.shape).astype(np.uint8)
    inner_mask.setflags(write=True)
    
    # Loop through objects in image
    for annotation in im_data.iter_annotations():
        cell_mask = annotation.mask.draw(cell_mask, alpha=1, color='#ffffff') # cumulative
        tmp_cell_mask = annotation.mask.draw(cell_mask, alpha=1, color='#ffffff') # single
        
        tmp_cell_mask = cv2.cvtColor(tmp_cell_mask, cv2.COLOR_BGR2GRAY) # binary (0 - 255)
        tmp_cell_mask = np.pad(tmp_cell_mask, (pad,pad), mode='edge') # padded
        
        tmp_border_mask = np.zeros(tmp_cell_mask.shape) # base for single border
        
        contours, _ = cv2.findContours(tmp_cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get outline
        cv2.drawContours(tmp_border_mask, contours, -1, 255, thickness) # draw single outline
        cv2.drawContours(border_mask, contours, -1, 255, thickness) # add outline to cumulative border mask
        
        tmp_inner_mask = tmp_cell_mask==255 # boolean so i can add to cumulative mask with OR
        tmp_inner_mask = cv2.erode(tmp_inner_mask.astype(np.uint8),np.ones((gap,gap))) # erode
        inner_mask = inner_mask | tmp_inner_mask # add to cumulative inner mask
        
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY) # RGB -> binary mask
    border_mask = border_mask[pad:-pad, pad:-pad] # crop padded border mask
    inner_mask = 255*(inner_mask[pad:-pad, pad:-pad]).astype(np.uint8) # boolean -> 255 binary mask
    
    return cell_mask, border_mask, inner_mask

def read_coco_file(coco_filepath):
    '''
    Read a COCO file and return a Python dictionary.

    Args:
        coco_filepath (string): Path (relative or absolute) to the COCO file

    Returns:
        (dict) Dictionary containing the COCO file data
    '''
    with open(coco_filepath) as f:
        return json.load(f)

def coco_to_masks(cocodict, test=True):
    '''
    Creates masks from a COCO-structured dictionary.

    Takes a dictionary (in COCO JSON structure, defining a series of images and the 
    borders of all the objects contained in each) and produces 3 masks (as one 3-channel 
    image--TODO) - one of the objects as defined by the COCO file, one of thickened 
    object outlines, and one of the inner parts of the objects (the inner mask of an 
    object is an erosion of the object mask).

    It is assumed that the images in the COCO file are square. Masks will not be correctly 
    produced if images are not square.

    The function can be run in test mode, which stops after the first image, by setting 
    test=True (default). 

    Args:
        coco_json (dict): COCO-fomatted dictionary
        test (boolean): Run in test mode if True (default is True)
    
    Returns:
        ndarray: Array of the masks for all the images, each with shape 3xImage_shape--TODO
    '''
    cocodict = imantics.Dataset.from_coco(cocodict) # convert to imantics type

    for image in cocodict.iter_images(): # for each image
        cells, borders, inners = draw_masks(image) # TODO: single 3-channel image instead of 3 separate ims
        
        if test:
            break
    
    return cells, borders, inners