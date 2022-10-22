# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:05:50 2022

@author: james
"""

from skimage import io, filters
import numpy
import time

rose = numpy.int64(io.imread("rose.tif", as_gray=True))
aztec = numpy.int64(io.imread("Capture.png", as_gray=True))

gNine = numpy.array(
[[0.0007, 0.0017, 0.0033, 0.0048, 0.0054, 0.0048, 0.0033, 0.0017, 0.0007],
[0.0017, 0.0042, 0.0078, 0.0114, 0.0129, 0.0114, 0.0078, 0.0042, 0.0017],
[0.0033, 0.0078, 0.0146, 0.0213, 0.0241, 0.0213, 0.0146, 0.0078, 0.0033],
[0.0048, 0.0114, 0.0213, 0.0310, 0.0351, 0.0310, 0.0213, 0.0114, 0.0048],
[0.0054, 0.0129, 0.0241, 0.0351, 0.0398, 0.0351, 0.0241, 0.0129, 0.0054],
[0.0048, 0.0114, 0.0213, 0.0310, 0.0351, 0.0310, 0.0213, 0.0114 ,0.0048],
[0.0033, 0.0078, 0.0146, 0.0213, 0.0241, 0.0213, 0.0146, 0.0078, 0.0033],
[0.0017, 0.0042, 0.0078, 0.0114, 0.0129, 0.0114, 0.0078, 0.0042, 0.0017],
[0.0007, 0.0017, 0.0033, 0.0048, 0.0054, 0.0048, 0.0033, 0.0017, 0.0007]])

def briefDescriptors(image, keypoints, patchsize, sigma=2):
        #blurredImage = guassianBlur(image)        
        blurredImage = filters.gaussian(image, sigma, truncate=4.5) #is just faster than the one i wrote and gives similar results
        

def briefTest(intensityX, intensityY, is_colour = False):
    redWeight = 0.2 #weights used to get intensity from rgb image
    blueWeight = 0.5
    greenWeight = 0.3
    if is_colour:
        ix = redWeight * intensityX[0] + blueWeight * intensityX[1] + greenWeight * intensityX[2]
        iy = redWeight * intensityY[0] + blueWeight * intensityY[1] + greenWeight * intensityY[2]
        if iy > ix:
            return 1
        else:
            return 0
    else:
        if intensityY > intensityX:
            return 1
        else:
            return 0


def findFeatures(image):
    keypoints = []
    st = time.time()
    for row in range(3, len(image) - 4):
        for col in range(3, len(image[0]) - 4):
                findNineContPixels(image, keypoints, (row, col))
    print(time.time() - st)
    return keypoints, len(keypoints)


def findNineContPixels(image, keypoints, point):
    '''corner detection algorithm, FAST-9, uses a 16 pixel circle around the current pixel,
    and then tries to find a 9 continuos of either strictly darker or lighter pixels, which 
    classifies as a corner'''
    
    pointInt = image[point[0], point[1]]
    n = 12
    threshold = 30

    
    pointMax = pointInt + threshold
    pointMin = pointInt - threshold    
    
    points = [
        image[point[0] - 3, point[1]], ####
        image[point[0] - 3, point[1] + 1], 
        image[point[0] - 2, point[1] + 2], 
        image[point[0] - 1, point[1] + 3],
        image[point[0], point[1] + 3], ##
        image[point[0] + 1, point[1] + 3], 
        image[point[0] + 2, point[1] + 2], 
        image[point[0] + 3, point[1] + 1], 
        image[point[0] + 3, point[1]], ####
        image[point[0] + 3, point[1] - 1], 
        image[point[0] + 2, point[1] - 2], 
        image[point[0] + 1, point[1] - 3], 
        image[point[0], point[1] - 3], ##
        image[point[0] - 1, point[1] - 3], 
        image[point[0] - 2, point[1] - 2], 
        image[point[0] - 3, point[1] + 1]
        ]
    
    #print(pointInt, pointMin, pointMax, points)
    
    #find nine contigous points
    for i in range(len(points)):
        
        #fast test - on test image roses, shows
        if abs(points[0] - pointInt) < threshold and abs(points[8] - pointInt) < threshold:
            #print("1, 9")
           break
        if abs(points[4] - pointInt) < threshold and abs(points[12] - pointInt) < threshold:
            #print("5, 13")
            break
        
        
        #print()
        intensity = 0 # 1 == darker, 2 == brighter
        corner = True
        #print(points[i])
        if points[i] > pointMax:
            intensity = 1
        elif points[i] < pointMin:
            intensity = 2
        else:
            #print("next")
            continue

        for j in range(1, 9): #check next 8 pixels
            #print("comp:", points[((i + j) % len(points))])
            if intensity == 1: #if lighter
                if points[((i + j) % len(points))] < pointMax:
                    #print("yes")
                    corner = False #is not corner if not 9 contigious
                    break
            elif intensity == 2: # if darker
                if points[((i + j) % len(points))] > pointMin:
                    #print("no")
                    corner = False
                    break
        
        if corner:
            #print("corner", pointInt, pointMax, pointMin, intensity, points)
            keypoints.append([point, intensity])
            break
           
    return keypoints

def harrisCornerMeasure(keypoints):
    '''orders the kleypoints with the Harris corner measure algorithm
    '''
    orderedKeypoints = numpy.empty(keypoints.size())
    
    return orderedKeypoints





def guassianBlur(image, filter=gNine, isgrey=True):
    st = time.time()
    blurredImage = numpy.zeros((len(image), len(image[0])))
    filterSize = len(filter) #assume odd sqaure filter
    for y in range(len(image)):
        for x in range(len(image[0])):
            #print(x, y)
            cutOut = guassianGrid(image, filterSize, x, y)
            blurredImage[y,x] = numpy.round(numpy.sum(numpy.multiply(cutOut, filter)))
            blurredImage = numpy.uint8(blurredImage)
    print(time.time() - st)
    return blurredImage
         

def guassianGrid(image, filterSize, x, y):
    mid = int((filterSize - 1) / 2)
    #print(mid, x+mid, y+mid)
    cols = list(range(x-mid, x+mid+1))
    rows = list(range(y-mid, y+mid+1))
    #print(cols, rows)
    if (x - mid) < 0 and (y - mid) < 0:
        #top left corner
        #print("top left")
        imageCut= numpy.zeros((9,9))
        xzeros = yzeros = 0
        yi = xi = 0
        #print(cols)
        while cols[xi] < 0:
            xzeros = xzeros + 1
            xi = xi + 1
        while rows[yi] < 0:
            yzeros = yzeros + 1
            yi = yi + 1
        #print(cols, rows, xzeros, yzeros)
        for yi in range(yzeros, filterSize):
            for xi in range(xzeros, filterSize):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
        pass
    elif (x - mid) < 0 and (y + mid) > len(image) - 1:
        #print("bot left")
        imageCut= numpy.zeros((9,9))
        xzeros = yzeros = 0
        xi = yi = 0
        #print(cols)
        while cols[xi] < 0:
            xzeros = xzeros + 1
            xi = xi + 1
        while rows[yi] <= len(image) - 1:
            yzeros = yzeros + 1
            yi = yi + 1      
        yzeros = filterSize - yzeros
        #print(cols, rows, xzeros, yzeros)
        for yi in range(0, filterSize - yzeros):
            for xi in range(xzeros, filterSize):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
        pass
    elif (x + mid) > len(image[0]) - 1 and (y - mid) < 0:
        #top right corner
        #print("Top Right")
        imageCut= numpy.zeros((9,9))
        xzeros = yzeros = 0
        yi = xi = 0
        #print(cols)
        while cols[xi] <= len(image[0]) - 1:
            xzeros = xzeros + 1
            xi = xi + 1
        xzeros = filterSize - xzeros
        while rows[yi] < 0:
            yzeros = yzeros + 1
            yi = yi + 1
        #print(cols, rows, xzeros, yzeros)
        for yi in range(yzeros, filterSize):
            for xi in range(0, filterSize - xzeros):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
    elif (x + mid) > len(image[0]) - 1 and (y + mid) > len(image) - 1:
        #print("Bot Right")
        imageCut= numpy.zeros((9,9))
        xzeros = yzeros = 0
        yi = xi = 0
        #print(cols)
        while cols[xi] <= len(image[0]) - 1:
            xzeros = xzeros + 1
            xi = xi + 1
        xzeros = filterSize - xzeros
        while rows[yi] <= len(image) - 1:
            yzeros = yzeros + 1
            yi = yi + 1      
        yzeros = filterSize - yzeros
        #print(cols, rows, xzeros, yzeros)
        for yi in range(0, filterSize - yzeros):
            for xi in range(0, filterSize - xzeros):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
    elif (x - mid) < 0:
        #print("Left")
        imageCut= numpy.zeros((9,9))
        zeros = 0
        i = 0
        #print(cols)
        while cols[i] < 0:
            #print(cols[i], i)
            zeros = zeros + 1
            i = i + 1
        #print(cols, rows, zeros)
        for yi in range(filterSize):
            for xi in range(zeros, filterSize):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
    
    elif (x + mid) > len(image[0]) - 1:
        #print("Right")
        imageCut= numpy.zeros((9,9))
        zeros = 0
        i = 0
        #print(cols)
        while cols[i] <= len(image[0]) - 1:
            zeros = zeros + 1
            i = i + 1
        zeros = filterSize - zeros
        #print(cols, rows, zeros)
        for yi in range(filterSize):
            for xi in range(0, filterSize-zeros):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
    
    elif (y - mid) < 0:
        #print("Top")
        imageCut= numpy.zeros((9,9))
        zeros = 0
        i = 0
        #print(cols)
        while rows[i] < 0:
            zeros = zeros + 1
            i = i + 1
        #print(cols, rows, zeros)
        for yi in range(zeros, filterSize):
            for xi in range(filterSize):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
    elif (y + mid) > len(image) - 1:
        #print("Right")
        imageCut= numpy.zeros((9,9))
        zeros = 0
        i = 0
        #print(cols)
        while rows[i] <= len(image) - 1:
            #print(rows[i], i)
            zeros = zeros + 1
            i = i + 1
        zeros = filterSize - zeros
        #print(cols, rows, zeros)
        for yi in range(0, filterSize - zeros):
            for xi in range(filterSize):
                imageCut[yi, xi] = image[rows[yi], cols[xi]]
        return imageCut
    else:
        #middle (all in image)
        #print("middle")
        return image[numpy.ix_(rows, cols)]
