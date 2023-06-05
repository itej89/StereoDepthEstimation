import numpy as np

#Cotains calibration parameters of teh given datasets

class calibrationParameters_artroom:
    K_left = np.array([ 
       [1733.74, 0, 792.27],
       [0, 1733.74, 541.89],
       [0, 0, 1]  
    ])

    K_right = np.array([ 
       [1733.74, 0, 792.27],
       [0, 1733.74, 541.89],
       [0, 0, 1]  
    ])
    
    focal = K_left[0,0]
    doffs=0
    baseline=536.62
    width=1920
    height=1080
    ndisp=170
    vmin=55
    vmax=142


class calibrationParameters_chess:
    K_left = np.array([ 
       [1758.23, 0, 829.15],
       [ 0, 1758.23, 552.78],
       [ 0, 0, 1]  
    ])

    K_right = np.array([ 
       [1758.23, 0, 829.15],
       [ 0, 1758.23, 552.78],
       [ 0, 0, 1]  
    ])
    
    focal = K_left[0,0]
    doffs=0
    baseline=97.99
    width=1920
    height=1080
    ndisp=220
    vmin=65
    vmax=197


class calibrationParameters_ladder:
    K_left = np.array([ 
       [1734.16, 0, 333.49],
       [ 0, 1734.16, 958.05],
       [ 0, 0, 1]  
    ])

    K_right = np.array([ 
       [1734.16, 0, 333.49],
       [ 0, 1734.16, 958.05],
       [ 0, 0, 1]  
    ])
    
    focal = K_left[0,0]
    doffs=0
    baseline=228.38
    width=1920
    height=1080
    ndisp=110
    vmin=27
    vmax=85
