import cv2
import os
from PIL import Image
import numpy as np
import time
from datetime import datetime

def get_file_creation_time(file_path):
    # Получаем время последнего изменения файла
    timestamp = os.path.getmtime(file_path)
    # Конвертируем в читаемый формат (только дата)
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

def add_text_to_image(image, text):
    # Конвертируем изображение в формат для OpenCV
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image)
    
    # Настройки текста
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_color = (255, 255, 255)  # Белый цвет
    thickness = 2
    line_type = cv2.LINE_AA
    
    # Получаем размеры изображения
    height, width = img.shape[:2]
    
    # Получаем размер текста
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Позиция текста (по центру внизу)
    text_x = (width - text_size[0]) // 2
    text_y = height - 20
    
    # Добавляем черный фон для текста
    cv2.rectangle(img, (text_x - 10, text_y - text_size[1] - 10), 
                 (text_x + text_size[0] + 10, text_y + 10), 
                 (0, 0, 0), -1)
    
    # Добавляем текст
    cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                font_color, thickness, line_type)
    
    return img

def create_video_from_images(input_folder, output_video, fps=30, image_duration_ms=1000):
    # Получаем список всех изображений в папке
    images = [img for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # Сортируем изображения по имени
    
    if not images:
        print("В папке нет изображений!")
        return
    
    # Получаем размер первого изображения
    first_image = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = first_image.shape
    
    # Создаем объект VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Рассчитываем количество кадров для каждого изображения
    frames_per_image = int((image_duration_ms / 1000) * fps)
    
    # Добавляем каждое изображение в видео
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        
        # Получаем дату создания файла
        creation_time = get_file_creation_time(img_path)
        
        # Добавляем текст с датой на изображение
        frame_with_text = add_text_to_image(frame, creation_time)
        
        # Добавляем изображение несколько раз для нужной длительности
        for _ in range(frames_per_image):
            video.write(frame_with_text)
        
        print(f"Обработано изображение: {image} ({creation_time})")
    
    # Закрываем видео
    video.release()
    print(f"Видео успешно создано: {output_video}")

if __name__ == "__main__":
    input_folder = "gallery"  # Папка с фотографиями
    output_video = "output_video.mp4"  # Имя выходного видео файла
    fps = 30  # Количество кадров в секунду
    image_duration_ms = 1_000  # Длительность показа каждого изображения в миллисекундах
    
    create_video_from_images(input_folder, output_video, fps, image_duration_ms)
