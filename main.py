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

def debug_face_detection(image):
    # Загружаем классификатор для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Конвертируем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Обнаруживаем лица
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"Найдено лиц: {len(faces)}")
    
    # Создаем копию изображения для рисования
    debug_image = image.copy()
    
    for (x, y, w, h) in faces:
        # Вычисляем центр лица
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Рисуем красный крест
        cross_size = 20
        color = (0, 0, 255)  # Красный цвет в формате BGR
        thickness = 2
        
        # Горизонтальная линия
        cv2.line(debug_image, 
                (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), 
                color, thickness)
        
        # Вертикальная линия
        cv2.line(debug_image, 
                (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), 
                color, thickness)
        
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, thickness)
    
    return debug_image

def center_image_on_face(image):
    # Загружаем классификатор для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Конвертируем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Обнаруживаем лица
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Находим лицо с наибольшим размером
        largest_face = max(faces, key=lambda face: face[2] * face[3])  # face[2] - ширина, face[3] - высота
        (x, y, w, h) = largest_face
        
        # Получаем размеры изображения
        height, width = image.shape[:2]
        
        # Вычисляем центр лица
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Вычисляем смещение для центрирования
        shift_x = width // 2 - face_center_x
        shift_y = height // 2 - face_center_y
        
        # Создаем матрицу преобразования
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Применяем аффинное преобразование
        centered_image = cv2.warpAffine(image, M, (width, height))
        
        return centered_image
    else:
        # Если лицо не найдено, возвращаем оригинальное изображение
        return image

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
        
        # frame = debug_face_detection(frame)
        frame = center_image_on_face(frame)

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
