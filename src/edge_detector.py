import cv2
import numpy as np
from scipy import signal


class CannyEdgeDetector (object):
    def get_img_edges(self, img, sigma, low_threshold, high_threshold):
        grad_magnitute, grad_degree = self.get_mag_degree(img)
        supressed = self.non_maximal_supress(grad_magnitute, grad_degree)
        thresholded = self.double_threshold(supressed, low_threshold, high_threshold)
        output = self.edge_tracking(thresholded)
        
        return output
    
    @staticmethod
    def non_maximal_supress(img, gdegree):
        height, width = img.shape
    
        for x in range(0, width):
            for y in range(0, height):
                if x == 0 or y == height -1 or y == 0 or x == width -1:
                    img[y][x] = 0
                    continue
                direction = gdegree[y][x] % 4
                if direction == 0:
                    if img[y][x] <= img[y][x - 1] or img[y][x] <= img[y][x + 1]:
                        img[y][x] = 0
                if direction == 1:
                    if img[y][x] <= img[y - 1][x + 1] or img[y][x] <= img[y + 1][x - 1]:
                        img[y][x] = 0
                if direction == 2:
                    if img[y][x] <= img[y - 1][x] or img[y][x] <= img[y + 1][x]:
                        img[y][x] = 0
                if direction == 3:
                    if img[y][x] <= img[y - 1][x - 1] or img[y][x] <= img[y + 1][x + 1]:
                        img[y][x] = 0

        return img

    @staticmethod
    def double_threshold(img, low_threshold, high_threshold):
        img[np.where(img > high_threshold)] = 0
        img[np.where((img >= low_threshold) & (img <= high_threshold))] = 255
        img[np.where(img < low_threshold)] = 0

        return img

    @staticmethod
    def edge_tracking(img):
        height, width = img.shape

        for i in range(0, height):
            for j in range(0, width):
                if img[i][j] == 75:
                    if ((img[i + 1][j] == 255) or 
                        (img[i - 1][j] == 255) or 
                        (img[i][j + 1] == 255) or 
                        (img[i][j - 1] == 255) or 
                        (img[i + 1][j + 1] == 255) or 
                        (img[i - 1][j - 1] == 255)):
                        img[i][j] = 255
                    else:
                        img[i][j] = 0

        return img

    @staticmethod
    def get_mag_degree(img):
        Lx = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
        Ly = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
        G = np.array([[2, 4, 5, 4, 2],
                    [4, 9, 12, 9, 4],
                    [5, 12, 15, 12, 5],
                    [4, 9, 12, 9, 4],
                    [2, 4, 5, 4, 2]])
        G = G / 159
        G_x = signal.convolve2d(G, Lx, mode='same')
        G_y = signal.convolve2d(G, Ly, mode='same')
        magx = signal.convolve2d(img, G_x, mode='same')
        magy = signal.convolve2d(img, G_y, mode='same')

        mag = np.sqrt(magx**2 + magy**2)
        degree = np.arctan2(magy, magx)

        return mag, degree
    