import cv2
import numpy as np

from typing import List
from cv_lab1.src.edge_detector import CannyEdgeDetector

def get_foreground_mask(image_path: str) -> List[tuple]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    canny = CannyEdgeDetector()
    img_edges = canny.get_img_edges(img, 1, 25, 80)

    img = np.where(img_edges > 1, 255, 0).astype('uint8')
    pred_points = np.argwhere(img)
    pred_points[:, [1, 0]] = pred_points[:, [0, 1]]
    cv2.fillPoly(img, [pred_points], 255)
    pred_points = np.argwhere(img)

    return pred_points
