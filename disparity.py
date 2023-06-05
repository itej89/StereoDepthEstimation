from tqdm import tqdm
import numpy as np
import cv2 as cv
from PIL import Image

class disparity:

    def __init__(self, window, range) -> None:
        #Kernel size
        self.WINDOW = window

        #search range
        self.RANGE = range

    #Computes disparity for a given pair of rectified images
    def compute_disparity_map(self, left_img, right_img):
        left = np.asarray(Image.fromarray(left_img).convert('L'))
        right = np.asarray(Image.fromarray(right_img).convert('L'))

        w, h = left_img.shape[1],  left_img.shape[0]
        
        depth = np.zeros((w, h), np.uint8)
        depth.shape = h, w
        
        Bound = int(self.WINDOW / 2)
        
        #loop over each row
        for row in tqdm(range(Bound, h - Bound)):  

            #loop over each pixel in the row                  
            for col in range(Bound, w - Bound):
                best_cell = 0
                prev_error = 65534
                
                #look for matches around a given range my moving the kernel
                for offset in range(self.RANGE):               
                    ssd_error = 0                           

                    #compute error using the kernel at a given offset
                    for v in range(-Bound, Bound):
                        for u in range(-Bound, Bound):
                            dist = int(left[row+v, col+u]) - int(right[row+v, (col+u) - offset])  
                            ssd_error += dist * dist              
                    
                    #save best match
                    if ssd_error < prev_error:
                        prev_error = ssd_error
                        best_cell = offset

                #Compute disparity map              
                depth[row, col] = best_cell

        return depth
                                    