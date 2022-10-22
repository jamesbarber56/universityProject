
import numpy
import time

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
