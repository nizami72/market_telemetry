"""
This script converts timestamps in a CSV file
from a custom format (e.g., 'YYYY-MM-DD HH:MM:SS') to ISO 8601 format with a +00:00 offset,
while also adjusting the time by subtracting 4 hours.
"""
import os
import sys
from datetime import datetime, timedelta

def convert_timestamps(file_path):
    temp_path = file_path + ".tmp"

    print(f"⚙️ Запуск конвертации таймстемпов для файла: {file_path}...")

    try:
        with open(file_path, "r") as fin, open(temp_path, "w") as fout:
            for line in fin:
                if not line.strip():
                    fout.write(line)
                    continue

                parts = line.split(",", 1)
                timestamp_str = parts[0]

                try:
                    # Парсим старый формат. Если есть миллисекунды, отсекаем их.
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
        print("✅ Конвертация успешно завершена!")

    except FileNotFoundError:
        print(f"❌ Ошибка: Файл '{file_path}' не найден!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка при обработке: {e}")
        # Убираем временный файл, если скрипт упал
        if os.path.exists(temp_path):
            os.remove(temp_path)
        sys.exit(1)

if __name__ == "__main__":
    # Проверяем, передан ли аргумент с названием файла при запуске
    if len(sys.argv) < 2:
        print("❌ Ошибка запуска.")
        print("💡 Использование: python convert_time.py <путь_к_файлу.csv>")
        sys.exit(1)

    target_csv = sys.argv[1]
    convert_timestamps(target_csv)
