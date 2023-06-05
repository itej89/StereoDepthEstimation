import numpy as np
import cv2 as cv

class drawing:

    def drawPoints(self, img, pts):
        for pt in pts:
            cv.circle(img, tuple(pt), 5, (0, 255, 255), -1)

    def drawLines(self, img, lines):
        c= img.shape[1]
        for r in lines:
            if r[1] != 0:
                x0, y0 = map(int, [0, -r[2]/r[1]])
                x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
                cv.line(img, (x0, y0), (x1, y1), (255, 255, 0), 1)

    def resizeImg(self, Img, percent):
        scale_percent = percent # percent of original size
        width = int(Img.shape[1] * scale_percent / 100)
        height = int(Img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        return cv.resize(Img, dim, interpolation = cv.INTER_AREA)

    #Draw epipolar lines
    def showEpipolarLines(self, imgL, imgR, leftPts, rightPts, F, title):

        random_idx = np.random.choice(len(leftPts), 25)
        pts_left = np.asarray(leftPts[random_idx], dtype=np.int32)
        pts_right = np.asarray(rightPts[random_idx], dtype=np.int32)
        self.drawPoints(imgL, pts_left)
        self.drawPoints(imgR, pts_right)

        #Draw right image epipolar lines
        epilines_right = cv.computeCorrespondEpilines(pts_right.reshape(-1, 1, 2), 2, F)
        epilines_right = epilines_right.reshape(-1, 3)
        self.drawLines(imgL, epilines_right)

        #Draw left image epipolar lines
        epilines_left = cv.computeCorrespondEpilines(pts_left.reshape(-1, 1, 2), 1, F)
        epilines_left = epilines_left.reshape(-1, 3)
        self.drawLines(imgR, epilines_left)

        combined = np.column_stack((imgL, imgR))
        resized = self.resizeImg(combined, 50)
        cv.imshow(title, resized)
        cv.imwrite("./"+title.replace(":","").replace(" ","_")+".jpg", resized)
        return imgL, imgR

    #Show rectified images side-by-side
    def showRectified(self,leftImg, rightImg, title):

        combined = np.column_stack((leftImg, rightImg))

        cv.imshow(title, self.resizeImg(combined, 50))
        cv.imwrite("./"+title.replace(":","").replace(" ","_")+".jpg", combined)
        return leftImg, rightImg
    
    #Show heat map of the given image
    def showHeatMap(self, array, title):
        array = cv.applyColorMap(array, cv.COLORMAP_JET)
        cv.imshow(title,array)
        cv.imwrite("./"+title.replace(":","").replace(" ","_")+".jpg", array)
        return array
    
    #Show image at a given scale
    def showImage(self, Image, title, percent):  
        resized = Image
        if percent < 100:
            resized = self.resizeImg(Image, percent)
        cv.imshow(title, resized)
        cv.imwrite("./"+title.replace(":","").replace(" ","_")+".jpg", resized)