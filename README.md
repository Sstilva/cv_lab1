# Задача выделения бинарной маски переднего плана
## Описание
Для задачи выделения бинарной маски был использован алгоритм Canny и последующая заливка найденных границ

### Алгоритм Canny 
1. Сглаживание изображения с использованием фильтра Гаусса для уменьшения шума.
2. Вычисление градиента изображения.
3. Вычисление амплитуды и направления градиента.
4. Не-максимальное подавление для выделения локальных максимумов градиента.
5. Двойной пороговый метод для выделения потенциальных граничных пикселей.
6. Трассировка границ для связывания граничных пикселей в линии.

### Реализация
`canny_edge_detector.py` - Содержит класс `CannyEdgeDetector`, который реализует алгоритм Canny для обработки изображений. Метод `get_img_edges` принимает  на вход изображение, стандартное отклонение для фильтра Гаусса, порог порогового метода, а затем возвращает изображение с обнаруженными границами.  
`main.py` - Демонстрация использования алгоритма Canny для обнаружения границ на изображении. Этот файл загружает изображение, передает его на обработку классу `CannyEdgeDetector` и сохраняет результат.   

## Метрики
* Метрика точности: 75%;
* Метрика скорости: 3.5 (с/Мп)

## Алгоритм запуска 
**Локально:**
  * Распаковать содержимое архива `Lab1.zip` внутри папки `lab1`
  * `pip install -r requirements.txt`
  * В корне проекта выполнить `python -m lab1`
