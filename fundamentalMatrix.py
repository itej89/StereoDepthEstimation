import os
import numpy as np
import cv2 as cv

np.random.seed(12)

import matplotlib.pyplot as plt

#Problem1-----------------------------
class fundamentalMatrix:

    

    def match_features(self, im1, im2, title):
        #Convert images to gray scale---------------------
        im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        #-------------------------------------------------

        #Run SIFT to find features-------------------------------
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(im1_gray, None)
        kp2, des2 = sift.detectAndCompute(im2_gray, None)
        #--------------------------------------------------------------------


        # show key points----------------------------------------------------
        # imgKeyPts = cv.drawKeypoints(
        #     im1_gray, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow(title, imgKeyPts)
        # -----------------------------------------------------

        #Run knn matcher to find the matches----------------------------------------------------
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        #---------------------------------


        # Keep good matches: calculate distinctive image features
        matchesMask = [[0, 0] for i in range(len(matches))]
        good_matches = []
        leftPts = []
        rightPts = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good_matches.append(m)
                rightPts.append(kp2[m.trainIdx].pt)
                leftPts.append(kp1[m.queryIdx].pt)

        # Draw the keypoint matches between both pictures
        draw_params = dict(matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0),
                        matchesMask=matchesMask[300:500],
                        flags=cv.DrawMatchesFlags_DEFAULT)

        imgMatches = cv.drawMatchesKnn(
            im1_gray, kp1, im2_gray, kp2, matches[300:500], None, **draw_params)
        

        return np.int32(leftPts), np.int32(rightPts), imgMatches
        
    
    def find_fundamental_matrix(self, left, right):
        A = []
        #Construct Coefficient matrix
        for i in range(len(left)):
            ul = left[i][0]
            vl = left[i][1]
            ur = right[i][0]
            vr = right[i][1]

            row = [ul*ur, ul*vr, ul, vl*ur, vl*vr, vl, ur, vr, 1]
            # row = [ur*ul, ur*vl, ur, vr*ul, vr*vl, vr, ul, vl, 1]
            A.append(row)

        #Compute fundamental matrix from lowest eigen vector
        [U, S, Vt] = np.linalg.svd(A)

        least_eig = Vt[-1, :]

        F = least_eig.reshape((3,3))



    

        return F
    
    def getFundamentalMatrixRansac(self, left, right, error_threshold=0.3):
        left_pts = left
        right_pts = right
        #Find best fundamental matrix through Ransac algorithm-------------------------------------
        max_inlier_count = 0
        best_inlier_indices = []
        

        #FIND fundamental USING RANSAC----------------------------------------------------------------------------------
        #---------------------------------------------------------------------------
        itr=  0
        while itr<80:
            itr += 1

            #sample points
            #Chosen the value 8 through trial and error and realiability
            random_idx = np.random.choice(len(left_pts), 8)
            sample_left_pts = left_pts[random_idx]
            sample_right_pts = right_pts[random_idx]

            F = self.find_fundamental_matrix(sample_left_pts, sample_right_pts)
            inlier_count = 0
            inlier_indices = []

            #estimate inliers using the computed homography---------
            for i in range(len(left_pts)):
                X = left_pts[i].copy()
                X = X.reshape(3, 1).T

                #make a homogrnous coordinate
                P = right_pts[i].copy()
                P = P.reshape(3, 1)


                #Compute correspondance error
                Xerror =  X @ F @ P

                #count inliers
                if abs(Xerror) < error_threshold:
                    inlier_count +=1
                    inlier_indices.append(i)

            #record highest inliers found so far
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_inlier_indices = inlier_indices.copy()


        # print(f"num_matches = {len(best_inlier_indices)}")
        #compute final homography using the best inliers discovered
        F = self.find_fundamental_matrix(left_pts[best_inlier_indices], right_pts[best_inlier_indices])

        #-------------------------------------------------------------------------------

        return F, left_pts[best_inlier_indices], right_pts[best_inlier_indices]


    def makeRankTwo(self, F):

        [U, S, Vt] = np.linalg.svd(F)
        S[2] = 0
        S = np.diag(S)
        F = U @ S @ Vt

        F = F/F[2,2]

        return F



