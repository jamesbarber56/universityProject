# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:05:50 2022

@author: james
"""

from skimage import io, filters, color, draw, exposure
from gmpy2 import xmpz, hamdist #allows efficent bitstring manipulation
import numpy
import time

rose = io.imread("rose.tif", as_gray=True)
aztec = io.imread("Capture.png", as_gray=True)
car = io.imread("car.jpg")

wallOne = io.imread("wall/img1.ppm")
wallTwo = io.imread("wall/img2.ppm")


def imageToGrey(img):
    img = color.rgb2gray(img)
    img = exposure.rescale_intensity(img, out_range=(0,255))
    img = numpy.int64(numpy.uint8(img))
    return img


def briefFeatures(image, threshold=30):
    image = imageToGrey(image) #image with whole values
    keypoints = getFeatures(image, threshold=threshold)
    keypoints = briefDescriptors(image, keypoints)
    
    return keypoints

def getTestPairs(patchsize, sigma):
    maxPatch = numpy.int64(patchsize / 2)
    testPairs = numpy.around(numpy.random.normal(0, sigma, (256, 4)))
    testPairs = numpy.int64(testPairs)
    for i in range(len(testPairs)):
        for j in range(0,4):
            if testPairs[i][j] > maxPatch:
                testPairs[i][j] = maxPatch
            elif testPairs[i][j] < -maxPatch:
                testPairs[i][j] = -maxPatch
    return testPairs

def briefDescriptors(image, keypoints, patchSize=31, matching = False, testPairs = []):
    image = imageToGrey(image)
    st = time.time()
    #blurredImage = guassianBlur(image)   
    sigma = numpy.sqrt(1/25 * patchSize ** 2)
    blurredImage = filters.gaussian(image, sigma, truncate=4.5) #is just faster than the one i wrote and gives similar results
    blurredImage = exposure.rescale_intensity(blurredImage, out_range=(0,255))
    blurredImage = numpy.int64(numpy.uint8(blurredImage)) 
    
    if not matching:
        testPairs = getTestPairs(patchSize, sigma)
        
    #brief 32 - 256 test patches / bitstring
    
    maxPairRow = len(image)
    maxPairCol = len(image[0])
    for i in range(0, len(keypoints)):
        (row, col) = keypoints[i]
        
        #gives rounded 256, 2 for the patch tests
        #print(testPairs[0])
        j = 0
        bitStringMpz = xmpz(0)
        for pair in testPairs:
            xPair = [row + pair[0], col + pair[1]]
            yPair = [row + pair[2], col + pair[3]]
            
            if xPair[0] < 0:
                xPair[0] = 0
            elif xPair[0] >= maxPairRow:
                xPair[0] = maxPairRow - 1
            if xPair[1] < 0:
                xPair[1] = 0
            elif xPair[1] >= maxPairCol:
                xPair[1] = maxPairCol - 1
                
            if yPair[0] < 0:
                yPair[0] = 0
            elif yPair[0] >= maxPairRow:
                yPair[0] = maxPairRow - 1
            if yPair[1] < 0:
                yPair[1] = 0
            elif yPair[1] >= maxPairCol:
                yPair[1] = maxPairCol - 1   
    
            #print(blurredImage[xPair[0]][xPair[1]])
            bit = briefTest(blurredImage[xPair[0]][xPair[1]], blurredImage[yPair[0]][yPair[1]])
            #bitString = bitString + bit
            bitStringMpz[j] = bit
            j = j + 1
        #print(bitString)
        #print(j)
        keypoints[i] = (row, col, bitStringMpz)
        #keypoints[i] = (row, col, bitString)
    print("BRIEF time: ",time.time() - st)
    return (keypoints, testPairs)

def briefTest(intensityX, intensityY):
    if intensityY > intensityX:
        return 1
    else:
        return 0


def matching(img1, img2, patchSize=31, limit=60, threshold=30):
    sigma = numpy.sqrt(1/25 * patchSize ** 2)  
    pairs = getTestPairs(patchSize, sigma)
    
    matches = [] #tuples of two features that match
    
    imageOne = imageToGrey(img1) #image with whole values
    imageTwo = imageToGrey(img2) #image with whole values
    
    imageOneKeypoints = getFeatures(imageOne, threshold=threshold)
    imageTwoKeypoints = getFeatures(imageTwo, threshold=threshold)
    imageOneDiscriptors = briefDescriptors(imageOne, imageOneKeypoints, patchSize=patchSize, matching=True, testPairs=pairs)
    imageTwoDiscriptors = briefDescriptors(imageTwo, imageTwoKeypoints, patchSize=patchSize, matching=True, testPairs=pairs)
    
    #Greedy Nearest Neighbour with a limit
    st = time.time()
    for one in range(0,len(imageOneDiscriptors[0])):
        match = False
        currentDist = limit + 1
        for two in range(0, len(imageTwoDiscriptors[0])):
            distance = hamdist(imageOneDiscriptors[0][one][2], imageTwoDiscriptors[0][two][2])
            #print(currentDist, distance, limit)
            if distance < limit:
                #print("match")
                match = True
                if distance < currentDist:
                    #print("better match")
                    currentDist = distance
                    currentMatch = (imageTwoDiscriptors[0][two], two)
        if match:
            matches.append((imageOneDiscriptors[0][one], currentMatch[0]))
    print("Matching Time:", time.time() - st)
    print("Matches Found:", len(matches))
    drawMatchFeatures(img1, img2, imageOneKeypoints, imageTwoKeypoints, matches)
    return matches
    

def concatImages(imgOne, imgTwo):
    imgHeight = max(len(imgOne), len(imgTwo))
    imgWidth = max(len(imgOne[0]), len(imgTwo[0]))
    
    combinedImage = numpy.zeros((imgHeight, imgWidth*2, 3))
    #print(imgHeight, imgWidth, "\n", len(imgOne), len(imgOne[0]),"\n", len(imgTwo) , len(imgTwo[0]))
    
    if len(imgOne) < imgHeight:
        heightPad = imgHeight - len(imgOne)
        imgOne = numpy.pad(imgOne, [(0, heightPad), (0,0), (0,0)])
    if len(imgOne[0]) < imgWidth:
        widthPad = imgWidth - len(imgOne[0])
        imgOne = numpy.pad(imgOne, [(0, 0), (0,widthPad), (0,0)])
        
    if len(imgTwo) < imgHeight:
        heightPad = imgHeight - len(imgTwo)
        imgTwo = numpy.pad(imgTwo, [(0, heightPad), (0,0), (0,0)])
    if len(imgTwo[0]) < imgWidth:
        widthPad = imgWidth - len(imgTwo[0])
        imgTwo = numpy.pad(imgTwo, [(0, 0), (0,widthPad), (0,0)])
            
    combinedImage[numpy.ix_(range(0,imgHeight), range(0,imgWidth))] = imgOne[numpy.ix_(range(0,imgHeight), range(0,imgWidth))]  
    combinedImage[numpy.ix_(range(0,imgHeight), range(imgWidth,imgWidth*2))] = imgTwo[numpy.ix_(range(0,imgHeight), range(0,imgWidth))] 
        
    combinedImage = numpy.uint8(combinedImage)
    return combinedImage, imgWidth

def drawFeatures(img, keypoints):
    colourImage = color.gray2rgb(img)
    for keypoint in keypoints:
        #print(keypoint)
        rr, cc = draw.circle_perimeter(keypoint[0], keypoint[1], 3)
        colourImage[rr,cc] = [0,0,250]
    for keypoint in keypoints:
        colourImage[keypoint[0]][keypoint[1]] = [255,0,0]
    io.imshow(numpy.uint8(colourImage)) 
    io.imsave("FAST-Keypoints.jpg", colourImage, quality=100)

def drawMatchFeatures(imageOne, imageTwo, imageOneKeypoints, imgTwoKeypoints, matches):
    matchedImage, width = concatImages(imageOne, imageTwo)
    print(len(imgTwoKeypoints))         
     
    for keypoint in imageOneKeypoints:
        #print(keypoint)
        rr, cc = draw.circle_perimeter(keypoint[0], keypoint[1], 3)
        matchedImage[rr,cc] = [0,0,250]
        matchedImage[keypoint[0]][keypoint[1]] = [255,0,0]
    
    for keypoint in imgTwoKeypoints:
        #print(keypoint)
        rr, cc = draw.circle_perimeter(keypoint[0], (keypoint[1]+width), 3)
        matchedImage[rr,cc] = [0,0,250]
        matchedImage[keypoint[0]][keypoint[1]+width] = [255,0,0]
    
    
    for i in range(0, len(matches)):
        startRow, startCol, discriptor = matches[i][0]
        endRow, endCol, discriptor = matches[i][1]
        endCol = endCol + width
        rr, cc = draw.line(startRow, startCol, endRow, endCol)
        matchedImage[rr,cc] = [0,255,0]

    io.imsave("matching.jpg", matchedImage, quality=100)
    io.imshow(numpy.uint8(matchedImage))


def getFeatures(img, threshold = 30):
    image = imageToGrey(img)
    keypoints= findFeatures(image, 9, threshold)
    print("Features found: ", len(keypoints))
    finalKeypoints = nonMaximalSuppression(image, keypoints)
    drawFeatures(img, finalKeypoints)
    print("Final Features found: ", len(finalKeypoints))
    return finalKeypoints


def circle16Pixel(image, point):
    points = [
        image[point[0] - 3, point[1]], #### 1
        image[point[0] - 3, point[1] + 1], 
        image[point[0] - 2, point[1] + 2], ######
        image[point[0] - 1, point[1] + 3],
        image[point[0], point[1] + 3], ## 5
        image[point[0] + 1, point[1] + 3], 
        image[point[0] + 2, point[1] + 2], ###
        image[point[0] + 3, point[1] + 1], 
        image[point[0] + 3, point[1]], #### 9
        image[point[0] + 3, point[1] - 1], 
        image[point[0] + 2, point[1] - 2], ######
        image[point[0] + 1, point[1] - 3], 
        image[point[0], point[1] - 3], ## 13
        image[point[0] - 1, point[1] - 3], 
        image[point[0] - 2, point[1] - 2], ###
        image[point[0] - 3, point[1] + 1]
        ]
    return points


def findFeatures(image, n, threshold):
    keypoints = []
    st = time.time()
    for row in range(3, len(image) - 4):
        for col in range(3, len(image[0]) - 4):
                findNContPixels(image, keypoints, (row, col), n, threshold)
    print("Find feature time: ",time.time() - st)
    return keypoints


def findNContPixels(image, keypoints, point, N = 9, threshold=30):
    '''corner detection algorithm, FAST-n, uses a 16 pixel circle around the current pixel,
    and then tries to find a n continuos of either strictly darker or lighter pixels, which 
    classifies as a corner'''
    
    pointInt = image[point[0], point[1]]
    pointMax = pointInt + threshold
    pointMin = pointInt - threshold    
    points = circle16Pixel(image, point)
    
    contNine = numpy.array(range(N))

    for i in range(7):
        if abs(points[i] - pointInt) <= threshold and abs(points[i+8] - pointInt) <= threshold:
                #print("7, 15")
            return keypoints

    #find nine contigous points
    for i in range(len(points)):   
        #print()
        intensity = 0 # 1 == darker, 2 == brighter, 0 = Similar
        corner = True
        #print(points[i])
        if points[i] >= pointMax:
            intensity = 1
        elif points[i] <= pointMin:
            intensity = 2
        else:
            intensity = 0
            #print("next")
            continue
        
        contNine[0] = points[i]
        for j in range(1, N): #check next 8 pixels
            #print("comp:", points[((i + j) % len(points))])
            contNine[j] = points[((i + j) % len(points))]
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
            if intensity == 1:
                cornerStrength = min(contNine) - pointMax + threshold
            elif intensity == 2:
                cornerStrength = pointMin - max(contNine) + threshold
                
            #print("corner", pointInt, pointMax, pointMin, intensity, points)
            keypoints.append([point, intensity, cornerStrength])
            break
    return keypoints


def getNeighbours(image, point):
    (row, col) = point
    neighbours = numpy.array(range(8))
    neighbours[0] = image[row - 1][col - 1]
    neighbours[1] = image[row - 1][col]
    neighbours[2] = image[row - 1][col + 1]
    neighbours[3] = image[row][col + 1]
    neighbours[4] = image[row + 1][col + 1]
    neighbours[5] = image[row + 1][col]
    neighbours[6] = image[row + 1][col - 1]    
    neighbours[7] = image[row][col - 1]
    return neighbours


def nonMaximalSuppression(image, keypoints):
    suppressedKeypoints = []
    st = time.time()
    corners = numpy.zeros((len(image), len(image[0])))
    for i in range(len(keypoints)):
        corners[keypoints[i][0][0]][keypoints[i][0][1]] = keypoints[i][2]
        
    #preform non-maximal suppression on the features
    for row in range(3, len(image) - 4):
        for col in range(3, len(image[0]) - 4):

            if corners[row][col] != 0:
                neighbours = getNeighbours(corners, (row, col))
                maxStrength = True
                x = False
                
                for neighbour in neighbours:
                    if neighbour >= corners[row][col]:
                        maxStrength = False
                        corners[row][col] = 0
                        break
                if maxStrength:
                    suppressedKeypoints.append((row, col))
                
    
    print("Non Maximal Suppression time:", time.time() - st)
    return suppressedKeypoints

    
#unused code for the machine learning part of FAST


def harrisCornerMeasure(keypoints):
    '''orders the kleypoints with the Harris corner measure algorithm
    '''
    orderedKeypoints = numpy.empty(keypoints.size())
    
    return orderedKeypoints

def partitionImage(image, threshold, x):
    brighter = []
    darker = []
    similar = []
    
    for row in range(3, len(image) - 4):
        for col in range(3, len(image[0]) - 4):
            point = (row, col)
            pointInt = image[point[0], point[1]]
            points = circle16Pixel(image, (row,col))
            if points[x] >= pointInt + threshold:
                darker.append(point)
            elif points[x] > pointInt - threshold and points[x] < pointInt + threshold:
                similar.append(point)
            elif points[x] <= pointInt - threshold:
                brighter.append(point)
            else:
                print("error undefined - point:", point)
                
    return numpy.array([darker, similar, brighter])


def machineLearningQuickening(image, threshold, n):
    corners, ammount = findFeatures(rose, n, threshold)
    imageSize = len(image) * len(image[0])
    partitions = []
    for x in range(0,16):
        partitions.append(partitionImage(image, threshold, x))
    #entropy = (ammount + (imageSize - ammount)) logbase2 (ammount + (imageSize - ammount))
    #           - ammount logbase2 ammount - (imageSize - ammount) logbase2 (imageSize - ammount)