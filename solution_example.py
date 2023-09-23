from typing import List

import cv2
import numpy as np

import numpy
from scipy import signal

import cv2
import numpy as np

def nonMaximalSupress(img,gdegree):
    height, width = img.shape
    
    for x in range(0,width):
        for y in range(0,height):
            if x == 0 or y == height -1 or y == 0 or x == width -1:
                img[y][x] = 0
                continue
            direction = gdegree[y][x] % 4
            if direction == 0:
                if img[y][x] <= img[y][x-1] or img[y][x] <= img[y][x+1]:
                    img[y][x] = 0
            if direction == 1:
                if img[y][x] <= img[y-1][x+1] or img[y][x] <= img[y+1][x-1]:
                    img[y][x] = 0
            if direction == 2:
                if img[y][x] <= img[y-1][x] or img[y][x] <= img[y+1][x]:
                    img[y][x] = 0
            if direction == 3:
                if img[y][x] <= img[y-1][x-1] or img[y][x] <= img[y+1][x+1]:
                    img[y][x] = 0
    return img

def doubleThreshold(image, lowThreshold, highThreshold):
    image[numpy.where(image > highThreshold)] = 0
    image[numpy.where((image >= lowThreshold) & (image <= highThreshold))] = 255
    image[numpy.where(image < lowThreshold)] = 0
    return image

def edgeTracking(image):
    height, width = image.shape
    for i in range(0, height):
        for j in range(0, width):
            if image[i][j] == 75:
                if ((image[i+1][j] == 255) or (image[i - 1][j] == 255) or (image[i][j + 1] == 255) or (image[i][j - 1] == 255) or (image[i+1][j + 1] == 255) or (image[i-1][j - 1] == 255)):
                    image[i][j] = 255
                else:
                    image[i][j] = 0
    return image

def getMagDegree(img):
    Lx = np.array([[1,0,-1],
                   [2,0,-2],
                   [1,0,-1]])
    Ly = np.array([[1,2,1],
                   [0,0,0],
                   [-1,-2,-1]])
    G = np.array([[2,4,5,4,2],
                  [4,9,12,9,4],
                  [5,12,15,12,5],
                  [4,9,12,9,4],
                  [2,4,5,4,2]])
    G = G / 159
    G_x = signal.convolve2d(G, Lx, mode='same')
    G_y = signal.convolve2d(G, Ly, mode='same')
    magx = signal.convolve2d(img, G_x, mode='same', )
    magy = signal.convolve2d(img, G_y, mode='same')

    mag = np.sqrt(magx**2 + magy**2)
    degree = np.arctan2(magy,magx)

    return mag, degree

def cannyEdgeDetector(img, sigma, lowThreshold, highThreshold):

    gradientMagnitute, gradientdegree = getMagDegree(img)
    supressed = nonMaximalSupress(gradientMagnitute, gradientdegree)
    thresholded = doubleThreshold(supressed, lowThreshold, highThreshold)
    output = edgeTracking(thresholded)
    return output

if __name__ == '__main__':
    img = cv2.imread('images\Abyssinian_9.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lenna = cannyEdgeDetector(img, 1, 25, 80)

    img = np.where(lenna > 1,255,0).astype('uint8')
    pred_points = np.argwhere(img)
    pred_points[:, [1, 0]] = pred_points[:, [0, 1]]
    cv2.fillPoly(img, [pred_points], 255)
    cv2.imshow('img', img)
    cv2.waitKey(0)



def get_foreground_mask(image_path: str) -> List[tuple]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # your code next
    ###

    # example submission
    pred_points = np.argwhere(img)
    return pred_points