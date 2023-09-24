import cv2
import numpy as np
from scipy import signal


class CannyEdgeDetector(object):
    def get_img_edges(self, img, threshold):
        grad_magnitute, grad_degree = self.get_mag_degree(img)
        supressed = self.non_max_supression(grad_magnitute, grad_degree)
        thresholded = self.threshold(supressed, threshold)
        output = self.edge_tracking(thresholded)

        return output

    @staticmethod
    def non_max_supression(mag, gdegree):
        height, width = mag.shape
        suppresed = np.zeros_like(mag)

        for x in range(1, width - 1):
            for y in range(1, height - 1):

                # Проверяем соседние пиксели в зависимости от направления градиента
                # 8 соседей - значит шаг по углу pi/8 = 180 / 8
                if (180 * 1 / 8 <= gdegree[y][x] <= 180 * 3 / 8) or (180 * 9 / 8 < gdegree[y][x] <= 180 * 11 / 8):
                    first_neibour = mag[y - 1][x + 1]
                    second_neibour = mag[y + 1][x - 1]

                elif (180 * 3 / 8 < gdegree[y][x] <= 180 * 5 / 8) or (180 * 11 / 8 < gdegree[y][x] <= 180 * 13 / 8):
                    first_neibour = mag[y - 1][x]
                    second_neibour = mag[y + 1][x]

                elif (180 * 5 / 8 < gdegree[y][x] <= 180 * 7 / 8) or (180 * 13 / 8 < gdegree[y][x] <= 180 * 15 / 8):
                    first_neibour = mag[y - 1][x - 1]
                    second_neibour = mag[y + 1][x + 1]

                else:
                    first_neibour = mag[y - 1][x - 1]
                    second_neibour = mag[y + 1][x + 1]

                if mag[y][x] > first_neibour and mag[y][x] > second_neibour:
                    suppresed[y][x] = mag[y][x]

        return suppresed

    @staticmethod
    def threshold(img, threshold):
        img[np.where((img >= threshold))] = 255
        img[np.where(img < threshold)] = 0

        return img

    @staticmethod
    def edge_tracking(img, weak=125):

        height, width = img.shape
        # используем blob-analysis

        for i in range(0, height):
            for j in range(0, width):
                if img[i][j] == weak:
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
        img_blured = signal.convolve2d(img, G, mode='same')
        magx = signal.convolve2d(img_blured, Lx, mode='same')
        magy = signal.convolve2d(img_blured, Ly, mode='same')

        mag = np.sqrt(magx ** 2 + magy ** 2)
        degree = np.rad2deg(np.arctan2(magy, magx))

        return mag, degree
    