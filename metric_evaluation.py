import os

import cv2
import numpy as np
import pandas as pd

from .solution_example import get_foreground_mask


def evaluate_iou(image_dir: str, anno_dir: str) -> float:
    """
    Метод для расчёта mean IoU на датасете
    :param image_dir - каталог с фото для анализа
    :param anno_dir - каталог с аннотациями для расчета метрики
    :return значение mean IoU с сохранением оценки по каждому фото в csv-файл
    """

    iou_data = dict()
    for _, _, files in os.walk(image_dir):
        assert files is not None, 'no files read'
        for image_name in files:
            print(f'Processing file {image_name}')
            sample_name = image_name[:image_name.find('.jpg')]

            # acquiring ground truth mask data
            mask_true = cv2.imread(os.path.join(anno_dir, f'{sample_name}.png'))
            assert mask_true is not None, 'mask is None'
            true_points = cv2.cvtColor(mask_true, cv2.COLOR_BGR2GRAY)
            true_points[true_points != 2] = 1
            true_points[true_points == 2] = 0
            true_points = np.argwhere(true_points)
            true_points_set = set([tuple(x) for x in true_points])

            # acquiring predicted mask
            pred_points = get_foreground_mask(image_path=os.path.join(image_dir, image_name))
            assert pred_points is not None, 'pred_points is None'
            pred_points_set = set([tuple(x) for x in pred_points])

            # calculating IoU
            iou = len(true_points_set.intersection(pred_points_set)) / len(true_points_set.union(pred_points_set))

            image_names = iou_data.get('image_names', [])
            image_names.append(sample_name)
            iou_data['image_names'] = image_names

            iou_values = iou_data.get('iou_values', [])
            iou_values.append(iou)
            iou_data['iou_values'] = iou_values

        pd.DataFrame(data=iou_data).to_csv('./lab1/detailed_results.csv')
        return np.mean(iou_data['iou_values'])