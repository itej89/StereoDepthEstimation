import os

from drawing import *
from fundamentalMatrix import *
from essentialMatrix import *
from rectification import *
from calibrationParameters import *
from normalization import *
from disparity import *
from depth import *

np.set_printoptions(suppress = True)

if __name__ == "__main__":

    #Load image paths
    artroom_left = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data" ,"artroom" ,"im0.png")
    artroom_right = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data" ,"artroom" ,"im1.png")

    chess_left = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data" ,"chess" ,"im0.png")
    chess_right = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data" ,"chess" ,"im1.png")

    ladder_left = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data" ,"ladder" ,"im0.png")
    ladder_right = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data" ,"ladder" ,"im1.png")
    

    #Create objects for each functionality
    _drawing = drawing()
    _fundamentalMatrix = fundamentalMatrix()
    _essentialMatrix = essentialMatrix()
    _rectification = rectification()


    #Create Image sets to loop over
    ImageSets = {}
    ImageSets["ArtROOM"] = (artroom_left, artroom_right)
    ImageSets["Chess"] = (chess_left, chess_right)
    ImageSets["Ladder"] = (ladder_left, ladder_right)

    #Dictionary of RANSAC Thresholds for each image set
    RANSACThreshold = {}
    RANSACThreshold["ArtROOM"] = 0.003
    RANSACThreshold["Chess"] = 0.08
    RANSACThreshold["Ladder"] = 0.03

    #Dictionary of Calibration Parameters for each Iamge set
    CalibrationParameters = {}
    CalibrationParameters["ArtROOM"] = calibrationParameters_artroom
    CalibrationParameters["Chess"] = calibrationParameters_chess
    CalibrationParameters["Ladder"] = calibrationParameters_ladder

    #Loop over each image set
    for key in ImageSets:
        left_path = ImageSets[key][0]
        right_path = ImageSets[key][1]
        imgLeft = cv.imread(left_path)
        imgRight = cv.imread(right_path)


        #Match features-------------------------------------------------------------------------------------------------
        leftPts, rightPts, imgMatches = _fundamentalMatrix.match_features(imgLeft, imgRight, f"SIFT Features : {key}")
        _drawing.showImage(imgMatches, f"Feature matches : {key}", 50)
        #---------------------------------------------------------------------------------------------------------------

        #Normalize points-----------------------------------------------------------------------------------------------
        _normalization = normalization(leftPts, rightPts)
        norm_leftPts, norm_rightPts = _normalization.normalizeHomogenous(leftPts, rightPts)
        #---------------------------------------------------------------------------------------------------------------


        #Find Fundamenta matrix-----------------------------------------------------------------------------------------
        F, best_left_pts, best_right_pts  = _fundamentalMatrix.getFundamentalMatrixRansac(norm_leftPts, norm_rightPts, RANSACThreshold[key])
        leftPts, rightPts  = _normalization.unnormalizeHomogenous(best_left_pts, best_right_pts)
        F = _normalization.unnormalizeFundamentalMatrix(F)
        F = _fundamentalMatrix.makeRankTwo(F)
        #---------------------------------------------------------------------------------------------------------------


        #Find essential matrix and decompose it--------------------------------------------------------------------------
        E = _essentialMatrix.getEssentialMatrix(F, CalibrationParameters[key])
        C1, C2, C3, C4, R1, R2, R3, R4 = _essentialMatrix.getCameraPoses(E)
        #---------------------------------------------------------------------------------------------------------------


        #Compute epipolar lines--------------------------------------------------------------------------
        leftImgEpiLines, rightImgEpiLines =  _drawing.showEpipolarLines(imgLeft, imgRight, leftPts, rightPts, F , f"Epipolar lines for : {key}")

        print(f"\n\n-----------------------------------")
        print(f"\n\nFundamental matrix of {key} : \n{np.around(F,3)}")
        print(f"\nEssential matrix of {key} : \n{E}")
        print(f"\nRotaion and Translation of {key} : ")
        print(f"\nSolution 1 : \nC1  :{C1}, \nR1 : {R1}")
        print(f"\nSolution 2 : \nC2  :{C2}, \nR2 : {R2}")
        print(f"\nSolution 3 : \nC3  :{C3}, \nR3 : {R3}")
        print(f"\nSolution 4 : \nC4  :{C4}, \nR4 : {R4}")
        #-----------------------------------------------------------------------

        #Find Homography matrices------------------------------------------
        Hleft, Hright = _rectification.getRectificationMatrices(left_path, right_path, leftPts, rightPts, F)
        print(f"\nLeft Homography matrix of {key} : \n{Hleft}")
        print(f"\nRight Homography matrix of {key} : \n{Hright}")
        #-----------------------------------------------------------------

        #Image Rectification with epilines for visualization-----------------------
        leftImg_rectified, rightImg_rectified = _rectification.rectifyImages(
            leftImgEpiLines, rightImgEpiLines, Hleft, Hright)
        #-----------------------------------------------------------------------
        _drawing.showRectified(leftImg_rectified, rightImg_rectified, f"Rectified Image : {key}")


        #Image Rectification without epilines for depth computaiton-----------------------
        leftImg_rectified, rightImg_rectified = _rectification.rectifyImages(
            imgLeft, imgRight, Hleft, Hright)
        #-----------------------------------------------------------------------
        

        #down scale iamges for resonalble computation speed-----------------------
        leftImg_rectified = _drawing.resizeImg(leftImg_rectified, 50)
        rightImg_rectified = _drawing.resizeImg(rightImg_rectified, 50)
        #-----------------------------------------------------------------------

        #Find disparity--------------------------------------------------------------
        _disparity = disparity(6, 30)
        disparity_map = _disparity.compute_disparity_map(leftImg_rectified, rightImg_rectified)

        disparity_img = disparity_map * 255 / 30
        
        out_min = 0.0
        out_max = 255.0
        in_min = np.min(disparity_img)
        in_max = np.max(disparity_img)
        map_range = lambda x: (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min
        disparity_img = np.uint8(map_range(disparity_img))

        _drawing.showImage(disparity_img, f"Disparity gray map of : {key}", 100)
        _drawing.showHeatMap(disparity_img, f"Disparity color map of : {key}")
        #-----------------------------------------------------------------------



        #Find Depth--------------------------------------------------------------
        _depth = depth()
        depth_arr = _depth.computeDepth(disparity_map, CalibrationParameters[key])
        
        #display as depth map-----------------------
        depth_img = _depth.computeDepth(disparity_img, CalibrationParameters[key])
        depth_img = depth_img.astype(np.uint8)
        _drawing.showImage(depth_img, f"Depth gray map of : {key}", 100)
        _drawing.showHeatMap(depth_img, f"Depth color map of : {key}")
        #-------------------------------------------

        #-----------------------------------------------------------------------

 
    cv.waitKey(0)
    cv.destroyAllWindows()


    print("Stereo depth estimation has been completed.")