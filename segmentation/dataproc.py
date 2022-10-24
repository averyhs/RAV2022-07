# ================================================================== #
# Data processing                                                    #
#                                                                    #
# This file contains functions for preprocessing and postprocessing  #
# data for the NN.                                                   #
# ================================================================== #

import json

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