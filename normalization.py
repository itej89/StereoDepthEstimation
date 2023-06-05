import numpy as np

class normalization:


    #Computes nomalization matrices for given set of poitns
    def getNormalizationMatrix(self, points):

        x_mean = np.mean(points[:,0])
        y_mean = np.mean(points[:,1])

        x_std = np.std(points[:,0])
        y_std = np.std(points[:,1])

        Sx = 1/x_std
        Sy = 1/y_std

        dx = - Sx * x_mean
        dy = - Sy * y_mean


        T =  np.array([
            [Sx, 0, dx],
            [0, Sy, dy],
            [0,  0,  1]
        ]) 

        return T
    
    
    def __init__(self, leftPts, rightPts) -> None:
        
        self.Tleft  = self.getNormalizationMatrix(leftPts)
        self.Tright = self.getNormalizationMatrix(rightPts)

    #Normalize the poitns in homogenous coordinates
    def normalizeHomogenous(self, leftPts,rightPts):
        left_hom_pts = np.column_stack((leftPts, np.ones((leftPts.shape[0],1))))
        right_hom_pts = np.column_stack((rightPts, np.ones((rightPts.shape[0],1))))

        norm_leftPts =  ( self.Tleft  @ left_hom_pts.T  ).T
        norm_rightPts = ( self.Tright @ right_hom_pts.T ).T


        return norm_leftPts, norm_rightPts
    
    #Revert the normalized points back to image coordinates
    def unnormalizeHomogenous(self, leftPts, rightPts):

        leftPts = (np.linalg.inv(self.Tleft) @ leftPts.T).T
        rightPts = (np.linalg.inv(self.Tright) @ rightPts.T).T

        leftPts  = leftPts[:, :2]
        rightPts = rightPts[:, :2]

        return leftPts, rightPts
    
    #Convert fundamental matrix computed from normalized points back to image coordinates
    def unnormalizeFundamentalMatrix(self, F):
        return self.Tright.T @ F @ self.Tleft