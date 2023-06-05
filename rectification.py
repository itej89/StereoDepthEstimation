import numpy as np
import cv2 as cv

class rectification:

    #Fidn homography matrices for left and right images
    def getRectificationMatrices(self, leftPath, rightPath, leftPts, rightPts, F):
        leftImg = cv.imread(leftPath)
        rightImg = cv.imread(rightPath)

        h1, w1,_ = leftImg.shape
        h2, w2,_ = rightImg.shape
        status, HLeft, HRight = cv.stereoRectifyUncalibrated(
            np.float32(leftPts), np.float32(rightPts), F, imgSize=(w1, h1))
        print(f"Rectification status : {status}")
        HLeft = HLeft/HLeft[2,2]
        HRight = HRight/HRight[2,2]
        return HLeft, HRight
    
    #REctify given images
    def rectifyImages(self, leftImg, rightImg, HLeft, HRight):

        h1, w1,_  = leftImg.shape
        h2, w2,_ = rightImg.shape
        leftImg_rectified = cv.warpPerspective(leftImg, HLeft, (w1, h1))
        rightImg_rectified = cv.warpPerspective(rightImg, HRight, (w2, h2))

        return leftImg_rectified, rightImg_rectified