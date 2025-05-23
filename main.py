import os
from datetime import datetime

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips


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
        cv2.line(
            debug_image,
            (center_x - cross_size, center_y),
            (center_x + cross_size, center_y),
            color, thickness,
        )

        # Вертикальная линия
        cv2.line(
            debug_image,
            (center_x, center_y - cross_size),
            (center_x, center_y + cross_size),
            color, thickness,
        )

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
    cv2.rectangle(
        img, (text_x - 10, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0), -1,
    )

    # Добавляем текст
    cv2.putText(
        img, text, (text_x, text_y), font, font_scale,
        font_color, thickness, line_type,
    )

    return img


def crop_to_vertical(image):
    height, width = image.shape[:2]

    # Определяем желаемое соотношение сторон (9:16)
    target_ratio = 9 / 16

    # Вычисляем новую ширину, сохраняя высоту
    new_width = int(height * target_ratio)

    # Если текущая ширина меньше нужной, увеличиваем изображение
    if width < new_width:
        scale = new_width / width
        image = cv2.resize(image, (new_width, int(height * scale)))
        height, width = image.shape[:2]

    # Вычисляем координаты для обрезки по центру
    start_x = (width - new_width) // 2
    end_x = start_x + new_width

    # Обрезаем изображение
    cropped = image[:, start_x:end_x]

    return cropped


def add_audio_to_video(video_path, audio_path, output_path):
    """
    Добавляет аудио дорожку к видео файлу
    
    Args:
        video_path (str): Путь к видео файлу
        audio_path (str): Путь к аудио файлу
        output_path (str): Путь для сохранения итогового видео
    """
    try:
        # Загружаем видео и аудио
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Получаем длительность видео
        video_duration = video.duration

        # Если аудио длиннее видео, обрезаем его
        if audio.duration > video_duration:
            audio = audio.subclip(0, video_duration)
        # Если аудио короче видео, зацикливаем его
        elif audio.duration < video_duration:
            # Вычисляем сколько раз нужно повторить аудио
            loops = int(video_duration / audio.duration) + 1
            # Создаем список с повторяющимися аудио клипами
            audio_clips = [audio] * loops
            # Склеиваем все клипы в один
            audio = concatenate_audioclips(audio_clips).subclip(0, video_duration)

        # Добавляем аудио к видео
        final_video = video.set_audio(audio)

        # Сохраняем результат
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Освобождаем ресурсы
        video.close()
        audio.close()
        final_video.close()

        print(f"Видео с музыкой успешно создано: {output_path}")

        # Удаляем временный файл без музыки
        if os.path.exists(video_path) and video_path != output_path:
            os.remove(video_path)
            print(f"Временный файл удален: {video_path}")

    except Exception as e:
        print(f"Ошибка при добавлении аудио: {e}")


def apply_vintage_effects(image, frame_number=0):
    """
    Применяет винтажные эффекты к изображению

    Args:
        image: Входное изображение
        frame_number: Номер кадра для создания мерцания

    Returns:
        Изображение с винтажными эффектами
    """
    # Создаем копию изображения
    vintage_img = image.copy()
    height, width = vintage_img.shape[:2]

    # 1. Эффект сепии
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(vintage_img, cv2.COLOR_BGR2GRAY)

    # Создаем сепию
    sepia = np.zeros((height, width, 3), dtype=np.uint8)
    sepia[:, :, 0] = gray * 0.272  # Синий канал
    sepia[:, :, 1] = gray * 0.534  # Зеленый канал
    sepia[:, :, 2] = gray * 0.769  # Красный канал

    # 2. Добавляем зернистость пленки
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    sepia = cv2.add(sepia, noise)

    # 3. Виньетирование (затемнение по краям)
    # Создаем маску для виньетирования
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Создаем градиент виньетки
    vignette = 1 - (distance / max_distance) * 0.6
    vignette = np.clip(vignette, 0.3, 1.0)

    # Применяем виньетку
    for channel in range(3):
        sepia[:, :, channel] = sepia[:, :, channel] * vignette

    # 4. Мерцание яркости (как в старом кино)
    flicker_intensity = 0.1 + 0.05 * np.sin(frame_number * 0.3)
    sepia = cv2.convertScaleAbs(sepia, alpha=1 + flicker_intensity, beta=0)

    # 5. Добавляем царапины (случайно)
    if np.random.random() < 0.1:  # 10% шанс появления царапины
        scratch_x = np.random.randint(0, width)
        scratch_length = np.random.randint(height // 4, height // 2)
        scratch_y = np.random.randint(0, height - scratch_length)

        # Рисуем вертикальную царапину
        cv2.line(
            sepia, (scratch_x, scratch_y),
            (scratch_x + np.random.randint(-5, 5), scratch_y + scratch_length),
            (200, 200, 200), 1,
        )

    # 6. Уменьшаем контрастность для более мягкого вида
    sepia = cv2.convertScaleAbs(sepia, alpha=0.9, beta=10)

    # 7. Добавляем легкое размытие для эффекта старой оптики
    sepia = cv2.GaussianBlur(sepia, (3, 3), 0.5)

    return sepia


def create_video_from_images(
    input_folder,
    output_video,
    fps=30,
    image_duration_ms=1000,
    video_format='mp4',
    audio_file=None,
    vintage_effects=False,
):
    images = [img for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if not images:
        print("В папке нет изображений!")
        return

    # Загружаем первое изображение для определения размера
    first_image = cv2.imread(os.path.join(input_folder, images[0]))

    # Центрируем и обрезаем первое изображение
    first_image = center_image_on_face(first_image)
    first_image = crop_to_vertical(first_image)

    # Получаем размеры после обрезки
    height, width = first_image.shape[:2]

    # Выбираем кодек в зависимости от формата
    if video_format.lower() == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 кодек
    elif video_format.lower() == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID кодек
    elif video_format.lower() == 'mov':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 кодек
    else:
        print(f"Неподдерживаемый формат видео: {video_format}. Используется MP4.")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # Добавляем расширение файла, если его нет
    if not output_video.lower().endswith(f'.{video_format.lower()}'):
        output_video = f"{output_video}.{video_format.lower()}"

    # Если указан аудио файл, создаем временное видео без звука
    temp_video_path = output_video
    if audio_file:
        temp_video_path = f"temp_{output_video}"

    video = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frames_per_image = int((image_duration_ms / 1000) * fps)
    frame_counter = 0

    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)

        # Центрируем изображение по лицу
        frame = center_image_on_face(frame)

        # Обрезаем изображение под вертикальный формат
        frame = crop_to_vertical(frame)

        creation_time = get_file_creation_time(img_path)
        frame_with_text = add_text_to_image(frame, creation_time)

        for frame_in_image in range(frames_per_image):
            # Применяем винтажные эффекты, если включены
            if vintage_effects:
                processed_frame = apply_vintage_effects(frame_with_text, frame_counter)
            else:
                processed_frame = frame_with_text

            video.write(processed_frame)
            frame_counter += 1

        effect_status = " (с винтажными эффектами)" if vintage_effects else ""
        print(f"Обработано изображение: {image} ({creation_time}){effect_status}")

    video.release()

    # Если указан аудио файл, добавляем его к видео
    if audio_file and os.path.exists(audio_file):
        print(f"Добавляем музыку из файла: {audio_file}")
        add_audio_to_video(temp_video_path, audio_file, output_video)
    else:
        if audio_file:
            print(f"Аудио файл не найден: {audio_file}. Создается видео без музыки.")
        print(f"Видео успешно создано: {output_video}")


if __name__ == "__main__":
    input_folder = "gallery"  # Папка с фотографиями
    output_video = "output_video"  # Имя выходного видео файла (без расширения)
    fps = 30  # Количество кадров в секунду
    image_duration_ms = 1000  # Длительность показа каждого изображения в миллисекундах
    video_format = 'mp4'  # Формат видео (mp4, avi, mov)
    audio_file = None  # Путь к аудио файлу (например: "music.mp3", "background.wav")
    vintage_effects = True  # Включить винтажные эффекты (сепия, виньетирование, зернистость, мерцание, царапины)

    # Примеры использования:
    # audio_file = "background_music.mp3"  # Для добавления музыки
    # audio_file = "song.wav"              # Поддерживаются различные форматы
    # vintage_effects = True               # Для создания ретро-эффектов
    # vintage_effects = False              # Для современного вида

    create_video_from_images(
        input_folder,
        output_video,
        fps,
        image_duration_ms,
        video_format,
        audio_file,
        vintage_effects,
    )
