# modules
import numpy as np
import cv2 as cv
#

# images
img1 = cv.imread('i2.jpg', 1)
img2 = cv.imread('i1.png', 1)
#

# av. hue of base
blue = img1[:, :, 0]
img1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
blur = cv.GaussianBlur(blue, (5, 5), 0)
trash, masker = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
mask = np.zeros(blue.shape, np.uint8) * 255

contours, trash = cv.findContours(
    masker, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(mask, contours, -1, (255), -1)
mcol = cv.mean(img1, mask=mask)[0]
#

# contour of img2
blue = img2[:, :, 0]
blur = cv.GaussianBlur(blue, (1, 1), 0)
trash, masker = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
masker = 255 - masker
erosion = cv.erode(masker, kernel, iterations=6)
dilation = cv.dilate(erosion, kernel, iterations=5)

contours, trash = cv.findContours(
    dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#

# sort contour


def spy(arr):
    return arr[0][0][0]


def spx(arr):
    return arr[0][0][1]


contours.sort(key=spx)
for i in range(5):
    contours[5 * i: 5 * i + 5] = sorted(contours[5 * i: 5 * i + 5], key=spy)
#

# proccessing
img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
table = []

for j in range(25):
    i = 0
    mask = np.zeros(img2[:, :, 0].shape, np.uint8)
    test = cv.drawContours(mask, contours, j, (255), -1)
    pixelpoints = np.transpose(np.nonzero(test))
    yel = red = 0

    for x in pixelpoints:
        if (img2[x[0], x[1], 0] > 20):
            red = yel = 0
            break
        if (img2[x[0], x[1], 0] > (mcol+3)):
            yel += 1
        else:
            red += 1

    if (yel or red):
        if (yel >= red*0.47058823529411764706):
            table.append(':|')
        else:
            table.append(':)')
    else:
        table.append('x<')
#

# results
img2 = cv.cvtColor(img2, cv.COLOR_HSV2BGR)

for i in range(25):
    font = cv.FONT_HERSHEY_SIMPLEX
    result = str(table[i])

    if (result == ':)'):
        col = (255, 255, 255)
    elif (result == ':|'):
        col = (0, 0, 0)
    else:
        col = (255, 0, 0)

    cv.putText(img2, result, tuple(
        contours[i][0][0]), font, 1, col, 2, cv.LINE_AA)

cv.imshow('1', img2)
cv.waitKey(0)
cv.destroyAllWindows()
#
