import os

import cv2.cv2 as cv2
import numpy as np
import math
import typer 

app = typer.Typer()

# Convert an image from BGR (Blue Green Red) to LAB (L Alpha Beta)
def BGRtoLab(img_in):
    split_src = cv2.split(img_in)
    L = 0.3811*split_src[2]+0.5783*split_src[1]+0.0402*split_src[0]
    M = 0.1967*split_src[2]+0.7244*split_src[1]+0.0782*split_src[0]
    S = 0.0241*split_src[2]+0.1288*split_src[1]+0.8444*split_src[0]

    L = np.where(L == 0.0, 1.0, L)
    M = np.where(M == 0.0, 1.0, M)
    S = np.where(S == 0.0, 1.0, S)

    _L = (1.0 / math.sqrt(3.0)) * ((1.0000 * np.log10(L)) + (1.0000 * np.log10(M)) + (1.0000 * np.log10(S)))
    Alpha = (1.0 / math.sqrt(6.0)) * ((1.0000 * np.log10(L)) + (1.0000 * np.log10(M)) + (-2.0000 * np.log10(S)))
    Beta = (1.0 / math.sqrt(2.0)) * ((1.0000 * np.log10(L)) + (-1.0000 * np.log10(M)) + (-0.0000 * np.log10(S)))

    img_out = cv2.merge((_L, Alpha, Beta))
    return img_out

# Convert an image from LAB (L Alpha Beta) to RGB (Red Green Blue) 
def LabtoRGB(img_in):
    split_src = cv2.split(img_in)
    _L = split_src[0]*1.7321
    Alpha = split_src[1]*2.4495
    Beta = split_src[2]*1.4142

    L = (0.33333*_L) + (0.16667 * Alpha) + (0.50000 * Beta)
    M = (0.33333 * _L) + (0.16667 * Alpha) + (-0.50000 * Beta)
    S = (0.33333 * _L) + (-0.33333 * Alpha) + (0.00000 * Beta)

    L = np.power(10, L)
    M = np.power(10, M)
    S = np.power(10, S)
    
    L = np.where(L == 1.0, 0.0, L)
    M = np.where(M == 1.0, 0.0, M)
    S = np.where(S == 1.0, 0.0, S)
    
    R = 4.36226*L-3.58076*M+0.1193*S
    G = -1.2186*L+2.3809*M-0.1624*S
    B = 0.0497*L-0.2439*M+1.2045*S

    img_out = cv2.merge((B, G, R))
    return img_out

# Compute the new color base on source and reference images 
def computeColor(src, mean_src, stdDev_src, mean_ref, stdDev_ref):
    val = src - mean_src
    coeff = stdDev_ref/stdDev_src
    result = coeff*val + mean_ref
    return result

# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

# Convert an image from float to uint8 
def FloatToUint8(image):
    image= np.clip(image, 0, 255)
    image=image.astype(np.uint8)
    return image

@app.command()
# Color transfer algorithm 
def colorTransfer(src:str, ref:str, output:str, gamma:float = 1.0):
    img_src = cv2.imread(src)
    img_ref = cv2.imread(ref)  
    
    img_src = BGRtoLab(img_src)
    img_ref = BGRtoLab(img_ref)

    mean_src, stddev_src = cv2.meanStdDev(img_src)
    mean_ref, stddev_ref = cv2.meanStdDev(img_ref)
    split_src = cv2.split(img_src)
    img_out = cv2.merge((computeColor(split_src[0], mean_src[0], stddev_src[0], mean_ref[0], stddev_ref[0]),
                         computeColor(split_src[1], mean_src[1], stddev_src[1], mean_ref[1], stddev_ref[1]),
                         computeColor(split_src[2], mean_src[2], stddev_src[2], mean_ref[2], stddev_ref[2])))

    img_out = LabtoRGB(img_out)
    img_out = FloatToUint8(img_out)

    cv2.imwrite(output, img_out)

@app.command()
def help():
    print ("help")


def main():
    app()

if __name__ == '__main__':
    main()