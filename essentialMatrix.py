import numpy as np
import cv2 as cv


class essentialMatrix:

    #Compute essential matrix
    # E = trans(Kl) * E * Kr
    def getEssentialMatrix(self, F, calibrationParameters):
        E = calibrationParameters.K_left.T @ F @ calibrationParameters.K_right
        return np.around(E, 6)
    

    #Decomposes given essential matrix into all possible mathematica solutions for rotation and translation matrices
    def getCameraPoses(self, E):
        [U, S, Vt] = np.linalg.svd(E)

        W = np.array([
            [0, -1, 0],[1, 0, 0],[0, 0, 1]
        ])

        #Compute tralation vectors
        C1 = U[:,2]; C2 = -U[:,2]; C3 = U[:,2]; C4 = -U[:,2]

        #Compute rotation matrices
        R1 = U @ W @ Vt; R2 = U @ W @ Vt
        R3 = U @ W.T @ Vt; R4 = U @ W.T @ Vt

        return C1, C2, C3, C4, R1, R2, R3, R4