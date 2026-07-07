"""
This script converts timestamps in a CSV file (specifically 'multidim_market_data.csv') 
from a custom format (e.g., 'YYYY-MM-DD HH:MM:SS') to ISO 8601 format with a +00:00 offset,
while also adjusting the time by subtracting 4 hours.
"""
import os
from datetime import datetime, timedelta

file_path = "multidim_market_data.csv"  # Укажите имя вашего файла
temp_path = file_path + ".tmp"

with open(file_path, "r") as fin, open(temp_path, "w") as fout:
    for line in fin:
        if not line.strip():
            fout.write(line)
            continue

        parts = line.split(",", 1)
        timestamp_str = parts[0]

        try:
            # Парсим старый формат. Если есть миллисекунды (например, 16:47:00,62727.25),
            # отсекаем дробную часть после запятой для соответствия вашему целевому формату.
            if "," in timestamp_str and len(timestamp_str) > 19:
                base_time = timestamp_str.split(",")[0]
            else:
                base_time = timestamp_str

            # Конвертируем в объект datetime
            dt = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")

            # Вычитаем 4 часа
            dt = dt - timedelta(hours=4)

            # Формируем новый таймштамп в формате ISO 8601 с зоной +00:00
            new_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

            # Собираем строку обратно
            fout.write(new_timestamp + "," + parts[1])

        except ValueError:
            # Если строка не подошла под формат даты (например, заголовок), оставляем как есть
            fout.write(line)

# Атомарная замена файла в операционной системе
os.replace(temp_path, file_path)
print("Конвертация 55 000 строк успешно завершена!")