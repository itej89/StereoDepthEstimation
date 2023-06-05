import numpy as np
import cv2 as cv

class depth:

    #Computes depth for a given disparity map and camera calibration parameters
    # Depth = baseline * focal / disparity
    def computeDepth(self, disparityArray, calibrationParams):

        one_by_disparity = 1/disparityArray
        depth = (calibrationParams.baseline*calibrationParams.focal)*one_by_disparity

        return depth
