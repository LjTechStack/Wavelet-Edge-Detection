import sys
import cv2
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as down


def main():
    lena = sys.argv[1]
    carriage = sys.argv[2]
    peppers = sys.argv[3]

    default_Lena = cv2.imread(lena, 0)  # extract image from file and make it gray scale
    default_Carriage = cv2.imread(carriage, 0)  # extract image from file and make it gray scale
    default_Peppers = cv2.imread(peppers, 0)  # extract image from file and make it gray scale

    daubechies_On_Pic(default_Lena, 'Daubechies Lena')
    daubechies_On_Pic(noise(default_Lena), 'Daubechies Noise Lena')
    daubechies_On_Pic(default_Carriage, 'Daubechies Carriage')
    daubechies_On_Pic(noise(default_Carriage), 'Daubechies Noise Carriage')
    daubechies_On_Pic(default_Peppers, 'Daubechies Peppers')
    daubechies_On_Pic(noise(default_Peppers), 'Daubechies Noise Peppers')


def daubechies_On_Pic(default_Pic, name):
    daubechies_Default_Pic_L1, daubechies_Default_Pic_H1 = daubechiesX(
        default_Pic)  # apply daubechies transformation on the x-axis, return low and high pass
    daubechies_Default_Pic_LL1, daubechies_Default_Pic_LH1 = daubechiesY(
        daubechies_Default_Pic_L1)  # apply daubechies thansformation on the y-axis of low pass, return lowlow and lowhigh pass
    daubechies_Default_Pic_HL1, daubechies_Default_Pic_HH1 = daubechiesY(
        daubechies_Default_Pic_H1)  # apply daubechies thansformation on the y-axis of high pass, return highlow and highhigh pass
    daubechies_Default_Pic_Total1 = combine(daubechies_Default_Pic_LH1, daubechies_Default_Pic_HL1,
                                            daubechies_Default_Pic_HH1)  # combine LH,HL,HH which represent diagonal, vertical, and horizontal edges

    daubechies_Default_Pic_L2, daubechies_Default_Pic_H2 = daubechiesX(
        daubechies_Default_Pic_LL1)  # apply daubechies transformation on the x-axis, return low and high pass
    daubechies_Default_Pic_LL2, daubechies_Default_Pic_LH2 = daubechiesY(
        daubechies_Default_Pic_L2)  # apply daubechies thansformation on the y-axis of low pass, return lowlow and lowhigh pass
    daubechies_Default_Pic_HL2, daubechies_Default_Pic_HH2 = daubechiesY(
        daubechies_Default_Pic_H2)  # apply daubechies thansformation on the y-axis of high pass, return highlow and highhigh pass
    daubechies_Default_Pic_Total2 = combine(daubechies_Default_Pic_LH2, daubechies_Default_Pic_HL2,
                                            daubechies_Default_Pic_HH2)

    daubechies_Default_Pic_L3, daubechies_Default_Pic_H3 = daubechiesX(
        daubechies_Default_Pic_LL2)  # apply daubechies transformation on the x-axis, return low and high pass
    daubechies_Default_Pic_LL3, daubechies_Default_Pic_LH3 = daubechiesY(
        daubechies_Default_Pic_L3)  # apply daubechies thansformation on the y-axis of low pass, return lowlow and lowhigh pass
    daubechies_Default_Pic_HL3, daubechies_Default_Pic_HH3 = daubechiesY(
        daubechies_Default_Pic_H3)  # apply daubechies thansformation on the y-axis of high pass, return highlow and highhigh pass
    daubechies_Default_Pic_Total3 = combine(daubechies_Default_Pic_LH3, daubechies_Default_Pic_HL3,
                                            daubechies_Default_Pic_HH3)

    daubechies_Default_Pic_L4, daubechies_Default_Pic_H4 = daubechiesX(
        daubechies_Default_Pic_LL3)  # apply daubechies transformation on the x-axis, return low and high pass
    daubechies_Default_Pic_LL4, daubechies_Default_Pic_LH4 = daubechiesY(
        daubechies_Default_Pic_L4)  # apply daubechies thansformation on the y-axis of low pass, return lowlow and lowhigh pass
    daubechies_Default_Pic_HL4, daubechies_Default_Pic_HH4 = daubechiesY(
        daubechies_Default_Pic_H4)  # apply daubechies thansformation on the y-axis of high pass, return highlow and highhigh pass
    daubechies_Default_Pic_Total4 = combine(daubechies_Default_Pic_LH4, daubechies_Default_Pic_HL4,
                                            daubechies_Default_Pic_HH4)

    daubechies_Default_Pic_Total34 = combineUp(daubechies_Default_Pic_Total4, daubechies_Default_Pic_Total3)
    scale(daubechies_Default_Pic_Total34)

    thresh(daubechies_Default_Pic_Total34, 0)
    daubechies_Default_Pic_Total234 = combineUp(daubechies_Default_Pic_Total34, daubechies_Default_Pic_Total2)
    scale(daubechies_Default_Pic_Total234)
    thresh(daubechies_Default_Pic_Total1, 0)

    daubechies_Default_Pic_Total1234 = combineUp(daubechies_Default_Pic_Total234, daubechies_Default_Pic_Total1)
    daubechies_Default_Pic_Total1234 = down.zoom(daubechies_Default_Pic_Total1234, [2, 2], order=0)
    scale(daubechies_Default_Pic_Total1234)
    # thresh(daubechies_Default_Pic_Total1234,0)

    fig = plt.figure(name)
    fig.add_subplot(221)  # create a plot
    plt.title(name)  # title of image
    plt.imshow(default_Pic, cmap='gray')  # plot the image
    fig.add_subplot(222)  # create a plot
    plt.title("Level 3&4")  # title of image
    plt.imshow(daubechies_Default_Pic_Total34, cmap='gray')  # plot the image
    fig.add_subplot(223)  # create a plot
    plt.title("Level 2&3&4")  # title of image
    plt.imshow(daubechies_Default_Pic_Total234, cmap='gray')  # plot the image
    fig.add_subplot(224)  # create a plot
    plt.title("Level 1&2&3&4")  # title of image
    plt.imshow(daubechies_Default_Pic_Total1234, cmap='gray')  # plot the image
    plt.show()  # show figures


def noise(default_Pic):
    noise_Pic = copy.deepcopy(default_Pic)  # make copy of Pic
    cv2.randn(noise_Pic, 0, 20)  # apply gaussian noise on an image and store in noise
    noise_Pic = impluse(noise_Pic + default_Pic, .001)  # apply SP noise on image that has gaussian noise
    return noise_Pic


def impluse(x, p):  # apply SP noise to the given image x by appling the probability p
    y = copy.deepcopy(x)  # create a copy of image being passed in
    thresh = 1 - p  # create a threshold point
    for a in range(y.shape[0]):  # foreach loop for x-axis
        for b in range(y.shape[1]):  # foreach loop for y-axis
            rand = random.random()  # randomize a number to easy probability of SP
            y[a][b] = 0 if rand < p else 255 if rand > thresh else x[a][
                b]  # add salt and pepper pixel depending on probability
    return y  # return new image after noise is added


def daubechiesX(x):  # apply x-axis transformation on to given image
    daubechies_Low = copy.deepcopy(x).astype(int)  # create a copy of image to gather low pass infomation
    daubechies_High = copy.deepcopy(x).astype(int)  # create a copy of image to gather high pass infomation
    daubechies_Low_Array = np.array(
        [-0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131])  # low pass array transformation
    daubechies_High_Array = np.array(
        [-0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226])  # high pass array transformation

    for a in range(x.shape[0]):  # foreach loop for x-axis
        for b in range(x.shape[1]):  # foreach loop for y-axis
            if b == x.shape[1] - 1:  # prevent array overflow
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a][b] +
                                        daubechies_Low_Array[2] * x[a][b] + daubechies_Low_Array[3] * x[a][
                                            b])  # apply the low array transformation for near array overflow
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a][b] +
                                         daubechies_High_Array[2] * x[a][b] + daubechies_High_Array[3] * x[a][
                                             b])  # apply the high array transformation for near arry overflow
            elif b == x.shape[1] - 2:  # prevent array overflow
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a][b + 1] +
                                        daubechies_Low_Array[2] * x[a][b + 1] + daubechies_Low_Array[3] * x[a][
                                            b + 1])  # apply the low array transformation for near array overflow
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a][b + 1] +
                                         daubechies_High_Array[2] * x[a][b + 1] + daubechies_High_Array[3] * x[a][
                                             b + 1])  # apply the high array transformation for near arry overflow
            elif b == x.shape[1] - 3:  # prevent array overflow
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a][b + 1] +
                                        daubechies_Low_Array[2] * x[a][b + 2] + daubechies_Low_Array[3] * x[a][
                                            b + 2])  # apply the low array transformation for near array overflow
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a][b + 1] +
                                         daubechies_High_Array[2] * x[a][b + 2] + daubechies_High_Array[3] * x[a][
                                             b + 2])  # apply the high array transformation for near arry overflow
            else:
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a][b + 1] +
                                        daubechies_Low_Array[2] * x[a][b + 2] + daubechies_Low_Array[3] * x[a][
                                            b + 3])  # apply the low array transformation
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a][b + 1] +
                                         daubechies_High_Array[2] * x[a][b + 2] + daubechies_High_Array[3] * x[a][
                                             b + 3])  # apply the high array transformation

    daubechies_Low = down.zoom(daubechies_Low, [.5, 1],
                               order=0)  # downsize the image by cutting it in half in the x-axis
    daubechies_High = down.zoom(daubechies_High, [.5, 1],
                                order=0)  # downsize the image by cutting it in half in the x-axis
    return daubechies_Low, daubechies_High  # return daubechies_Low and daubechies_High


def daubechiesY(x):  # apply y-axis transformation on to given image
    daubechies_Low = copy.deepcopy(x)  # create a copy of image to gather low pass infomation
    daubechies_High = copy.deepcopy(x)  # create a copy of image to gather high pass infomation
    daubechies_Low_Array = np.array(
        [-0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131])  # low pass array transformation
    daubechies_High_Array = np.array(
        [-0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226])  # high pass array transformation

    for a in range(x.shape[0]):  # foreach loop for x-axis
        for b in range(x.shape[1]):  # foreach loop for y-axis
            if a == x.shape[0] - 1:  # prevent array overflow
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a][b] +
                                        daubechies_Low_Array[2] * x[a][b] + daubechies_Low_Array[3] * x[a][
                                            b])  # apply the low array transformation for near array overflow
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a][b] +
                                         daubechies_High_Array[2] * x[a][b] + daubechies_High_Array[3] * x[a][
                                             b])  # apply the high array transformation for near arry overflow
            elif a == x.shape[0] - 2:  # prevent array overflow
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a + 1][b] +
                                        daubechies_Low_Array[2] * x[a + 1][b] + daubechies_Low_Array[3] * x[a + 1][
                                            b])  # apply the low array transformation for near array overflow
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a + 1][b] +
                                         daubechies_High_Array[2] * x[a + 1][b] + daubechies_High_Array[3] * x[a + 1][
                                             b])  # apply the high array transformation for near arry overflow
            elif a == x.shape[0] - 3:  # prevent array overflow
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a + 1][b] +
                                        daubechies_Low_Array[2] * x[a + 2][b] + daubechies_Low_Array[3] * x[a + 2][
                                            b])  # apply the low array transformation for near array overflow
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a + 1][b] +
                                         daubechies_High_Array[2] * x[a + 2][b] + daubechies_High_Array[3] * x[a + 2][
                                             b])  # apply the high array transformation for near arry overflow
            else:
                daubechies_Low[a][b] = (daubechies_Low_Array[0] * x[a][b] + daubechies_Low_Array[1] * x[a + 1][b] +
                                        daubechies_Low_Array[2] * x[a + 2][b] + daubechies_Low_Array[3] * x[a + 3][
                                            b])  # apply the low array transformation
                daubechies_High[a][b] = (daubechies_High_Array[0] * x[a][b] + daubechies_High_Array[1] * x[a + 1][b] +
                                         daubechies_High_Array[2] * x[a + 2][b] + daubechies_High_Array[3] * x[a + 3][
                                             b])  # apply the high array transformation

    daubechies_Low = down.zoom(daubechies_Low, [1, .5],
                               order=0)  # downsize the image by cutting it in half in the y-axis
    daubechies_High = down.zoom(daubechies_High, [1, .5],
                                order=0)  # downsize the image by cutting it in half in the y-axis
    return daubechies_Low, daubechies_High  # return daubechies_Low and daubechies_High


def combine(LH, HL, HH):  # combine LH,HL,HH which represent diagonal, vertical, and horizontal edges
    Total = copy.deepcopy(LH)  # create a placeholder for total
    for a in range(Total.shape[0]):  # foreach loop for x-axis
        for b in range(Total.shape[1]):  # foreach loop for y-axis
            Total[a][b] = math.sqrt((LH[a][b] * LH[a][b]) + (HL[a][b] * HL[a][b]) + (
                    HH[a][b] * HH[a][b]))  # store total of the 3 combine images into Total
    return Total  # return Total


def combineUp(small, large):  # combine the smaller image by upscaling to match the bigger image
    Total = copy.deepcopy(large)  # placeholder for final results
    temp = copy.deepcopy(small)  # placeholder for upscaling
    temp = down.zoom(temp, [2, 2], order=0)
    for a in range(Total.shape[0]):  # foreach loop for x-axis
        for b in range(Total.shape[1]):  # foreach loop for y-axis
            Total[a][b] = Total[a][b] * temp[a][b]
    return Total


def scale(x):  # x is image being scaled
    high = High_Value(x)  # obtain the high and low pixel value of the image for scaling
    if (high > 255):  # check if high is greater than 255 and downscale pixel
        scale = high / 255.0  # store scaling value
        for a in range(x.shape[0]):  # foreach loop for x-axis
            for b in range(x.shape[1]):  # foreach loop for y-axis
                x[a][b] = x[a][b] / scale  # downscale the value stored in image
    return x  # return as a uint8 type since the scaling has now been done


def High_Value(x):  # x is image being scanned for max and min value
    high = 0  # store high value
    for a in range(x.shape[0]):  # foreach loop for x-axis
        for b in range(x.shape[1]):  # foreach loop for y-axis
            if x[a][b] > high:  # find highest value
                high = x[a][b]  # store high value
    return high  # return low and high


def thresh(x, thres):  # x is image being threshold
    for a in range(x.shape[0]):
        for b in range(x.shape[1]):
            x[a][b] = test_value(x[a][b], thres)
    return x


def test_value(x, thres):
    return 0 if x <= thres else 255  # if x < 115 else 0


if __name__ == "__main__":
   main()
