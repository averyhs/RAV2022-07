import json
import numpy as np
import imantics
import cv2
from tqdm import tqdm
from patchify import patchify, unpatchify
from PIL import Image
from torch.utils.data import DataLoader

from params import patch_size, batch_size

class Data:
    def __init__(self, X, Y, patch=True):
        '''
        Initialize Data with data X and ground truth Y.

        Written for use with U-Net-Id style network.

        Assumptions:
        - X consists of 1 channel images
        - Y consists of 3 channel images (object, border, inner masks)
        - X and Y have same shape
        
        Args:
            X (array like): data (to be analyzed)
            Y (array like): ground truth
        '''
        if patch:
            # How many patches to break image into? Calculate number of rows and cols of grid of patches
            # We're assuming step=patch_size
            # (TODO: calculate patch size and step, and padding if necessary, such that images do not need to be cropped)
            sample_image = Y[0]
            image_width = sample_image.shape[1]
            image_height = sample_image.shape[0]
            self.num_patch_cols = image_width//patch_size  # number of patches that fit along width of image
            self.num_patch_rows = image_height//patch_size # number of patches that fit along height of image

        self.X = self.patch(X, 1) if patch else X
        self.Y = self.patch(Y, 3) if patch else Y

        self.dataloader = DataLoader(list(zip(self.X, self.Y)), batch_size=batch_size, shuffle=True)
    
    def patch(self, image_list, channels):
        '''
        Patchify a list of images.
        
        Patchify a list of images - the images will be cropped and split into patches, producing a 1D list of
        all 2D patches (ordered so images can be recovered). Supports multichannel and single channel images.
        
        Assumptions:  
        - All images have the same shape  
        - Patches are square  
        
        Process:  
        1. Crop to a size divisible by the patch size  
        2. Patchify  
        
        Args:
            image_list (np.array list): List of the image series (any list type). Each image must be a 2D numpy array.  
            channels (int): How many channels the image has
        
        Returns:
            np.array: List of patchified images 
        '''

        # Find closest size image divisible by patch size, assuming step=patch_size
        size_x = self.num_patch_cols*patch_size # image width (x len) that is divisible by patch_size
        size_y = self.num_patch_rows*patch_size # image height (y len) that is divisible by patch_size

        # Make empty array for patched images (6D)
        patched_image_list = np.empty(shape=[len(image_list), 
            self.num_patch_rows,self.num_patch_cols, patch_size,patch_size, channels])

        # Loop through images
        for image,idx in zip(image_list, range(len(image_list))):
            tmp_list = []
            
            # Process each channel of the image separately
            for c in range(channels):
                im = image if channels==1 else image[:,:,c]

                # Crop
                im = Image.fromarray(im) # get a PIL Image to crop
                im = im.crop((0, 0, size_x, size_y))
                
                # Patchify
                im = np.array(im) # back to np array for patching
                patches_im = patchify(im, (patch_size,patch_size), step=patch_size)
                
                # Save in temp list for recombination with other channels
                tmp_list.append(patches_im)
            
            # Recombine channels to single multichan image again
            patches_image = np.stack(tmp_list, axis=-1)
            
            # Add this patched multichannel image to the overall list
            patched_image_list[idx,:,:,:,:,:] = patches_image

        # Now the list still has the patches in the original image shape
        # - current list shape: [len(image_list), npat_x, npat_y, patch_size, patch_size, channels]
        # Reshape it to have a long list of patched images
        # - desired list shape: [len*npat_x*npat_y, patch_size,patch_size, channels]
        s = patched_image_list.shape
        patched_image_list = np.reshape(patched_image_list, (s[0]*s[1]*s[2], s[3],s[4], s[5]), order='C')
        
        # Squeeze for single channel images
        if channels==1:
            patched_image_list = np.squeeze(patched_image_list)

        return patched_image_list

    @classmethod
    def draw_masks(im_data):
        '''
        Create object mask, border mask, inner mask for an image (Intended for internal use).

        For a single image from a COCO-formatted dict, use the object masks to create a 
        border mask and an inner mask. The masks are returned as a single 3-channel image 
        (as a numpy array), with each mask drawn in a separate channel.

        It is assumed that the image is square (equal height and width dims). Masks will 
        not be correctly produced if image is not square.

        Args:
            im_data (imantics image): The mask image data from which to produce masks
        
        Returns:
            ndarray: Masks as 3-channel image (numpy array)
        '''
        # The border mask for each object is obtained by finding the edges of the object mask 
        # and drawing them thicker, extending both inside and outside the object. The inner 
        # mask for each object is obtained by performing an erosion operation on the object 
        # mask.

        # Borders: for borders, images will be expanded and then cropped
        # so that outlines will not be drawn along the edges (for objects cut off by image edge)
        # Set params:
        thickness = 3 # thickness of border to draw
        pad = thickness
        
        # Inners: inners are obtained by erosion of cell masks
        # The value of gap is the side length of a square erosion kernel
        # It should be just enough to separate touching objects
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

        # This stays blank as a base for when single object needs to be drawn
        blank_mask = np.zeros((im_data.height, im_data.width, 3)).astype(np.uint8)
        
        # Loop through objects in image
        for annotation in im_data.iter_annotations():
            cell_mask = annotation.mask.draw(cell_mask, alpha=1, color='#ffffff') # cumulative
            tmp_cell_mask = annotation.mask.draw(blank_mask, alpha=1, color='#ffffff') # single
            
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
        
        return np.stack((cell_mask, border_mask, inner_mask), axis=-1)

    @classmethod
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
    
    @classmethod
    def coco_to_masks(cls, cocodict, fout):
        '''
        Creates masks from a COCO-structured dictionary.

        Takes a dictionary (in COCO JSON structure) and for each image produces 3 masks (as 
        single 3-channel image) - one of the objects, one of thick object outlines, and one 
        of the inner parts of the objects.

        The 3-channel masks are all saved in one numpy .npz file, with file path given as arg.

        It is assumed that the images in the COCO file are square. Masks will not be correctly 
        produced if images are not square.

        The function can be run in test mode, which stops after the first image, by setting 
        test=True (default). 

        Args:
            coco_json (dict): COCO-fomatted dictionary
            fout (string): File path (name ending in .npz) to save the data
        
        Returns:
            ndarray: Array of the masks for all the images, each mask a 3 channel image
        '''
        cocodict = imantics.Dataset.from_coco(cocodict) # convert to imantics type

        masks = []
        for image in tqdm(cocodict.iter_images()): # for each image
            masks.append(cls.draw_masks(image))
        
        np.savez(fout, *masks)
        return masks
