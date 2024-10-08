import os
import cv2
import numpy as np
from scipy.ndimage import median_filter

def clahe(input, limit = 2.0, grid = (8,8)):
    print('Clahe=','input:',input,'/limit:',limit,'/grid:',grid)
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    dst = clahe.apply(img)
    return  dst


def unsharp(image, sigma = 5, strength = 0.8):
    print('unsharp=','input:',image,'/sigma:',sigma,'/strength:',strength)
    # Median filtering
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image_mf = median_filter(image, sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image - strength * lap

    # Saturate the pixels in either direction
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0

    return sharp

def Img_Setting(LOC,Save_Loc,Counter):
    dir = os.listdir(LOC)
    for _ in dir:
        print(LOC+'/'+_)
        img_R = unsharp(LOC+'/'+_, 5, 0.8)
        img_G = clahe(LOC+'/'+_, 4.0, (8, 8))
        img_B = clahe(LOC+'/'+_, 4.0, (10, 10))
        
        Output_Img = np.zeros([np.shape(img_R)[0],np.shape(img_R)[1],3])
        Output_Img[:,:,0] = img_R
        Output_Img[:,:,1] = img_G
        Output_Img[:,:,2] = img_B

        if Counter == 'Show':
            cv2.imshow("original", cv2.imread(LOC+'/'+_))
            img_gray = cv2.cvtColor(np.uint8(Output_Img), cv2.COLOR_BGR2GRAY)
            cv2.imshow('Img_PreProcessing_Color',np.uint8(Output_Img))
            cv2.imshow('Img_PreProcessing_Gray',img_gray)
            cv2.waitKey()
        elif Counter == 'ImgWrite':
            img_gray = cv2.cvtColor(np.uint8(Output_Img), cv2.COLOR_BGR2GRAY)
            print(Save_Loc+'/'+_)
            cv2.imwrite(Save_Loc+'/'+_,img_gray)
            print('----------------------------------------------------------')
        else:
            print('Error')

####### Setting (Counter:Write - Show or ImgWrite) #######
# Img_Path - Image location
Img_Path = '002_original_train-test_separate/train/img'
# Show - Show Preprosessing Color and Gray Image
# ImgWrite - Save the preprosessing image to the user specified location
Counter = 'ImgWrite'
# Save_Loc - Save Image location (Result_location = Folder/Directory location setting, Not Image Path)
Save_Loc = 'original_train-test_separate/lim4/train/img'

Img_Setting(Img_Path,Save_Loc,Counter)
