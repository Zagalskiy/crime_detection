import os
import torch
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

if torch.cuda.is_available():    
    print('CUDA is available. Version:', torch.version.cuda)
    print('CUDA device:', torch.cuda.get_device_name(0))
else:
    print('CUDA is not available.')

MODEL_NAME = 'yolov9_d7_e300'  # для хранения различных экспериментов

# Получаем путь к текущей папке /src
current_dir = os.path.dirname(os.path.abspath(__file__))  # для .py

# Находим папку проекта относительно папки /src
project_path = os.path.abspath(os.path.join(current_dir, os.pardir))
print(f"Путь к папке проекта {project_path}")

# Задаём путь к текущей моделе
model_path = os.path.join(current_dir, 'runs', 'detect', MODEL_NAME)

# Загрузка лучшей обученной модели
best_file = 'best.pt'
output_path = os.path.join(model_path, 'weights', best_file)
print(f"Путь к весам модели {output_path}")
model_yolov9_best = YOLO(output_path)
model_yolov9_best.info()  # для вывода информации о модели

# Визуализация предсказания на видео не из датасета
video_file = 'robbery.gif'  # robbery !
video_path = os.path.join(current_dir, video_file)
print(f"Путь к видео {video_path}")
results = model_yolov9_best.predict(video_path, conf=0.5, save=True, project=current_dir, name='Predicted')

torch.cuda.empty_cache()