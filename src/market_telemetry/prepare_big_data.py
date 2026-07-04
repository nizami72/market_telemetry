import os
import glob
import gzip
import pandas as pd

def compile_massive_dataset(output_path="../../data/multidim_market_data.csv"):
    # Путь к папке, где лежат активный CSV и .gz архивы от logrotate
    data_dir = "../../data/hetzner/"

    print("📦 Начинаю сборку монолитного датасета из архивов Linux...")
    all_chunks = []

    # ИСПРАВЛЕНИЕ: Переносим заголовок в переменную, чтобы вставить его в самый конец
    headers_list = ["timestamp", "price", "imbalance_5", "imbalance_20", "imbalance_50", "market_delta_10s", "trade_speed_10s", "label_next_price"]

    # 1. Сначала читаем текущий (недавно обрезанный) активный файл
    active_file = os.path.join(data_dir, "multidim_market_data.csv")
    if os.path.exists(active_file) and os.path.getsize(active_file) > 0:
        print(f"📖 Читаю активный файл: {active_file}")
        # Избегаем DtypeWarning и заставляем Pandas читать всё как строки/объекты перед склейкой
        df_active = pd.read_csv(active_file, header=None, dtype=str)
        all_chunks.append(df_active)

    # 2. Находим все заархивированные файлы .gz в этой директории
    archive_pattern = os.path.join(data_dir, "multidim_market_data.csv.*.gz")
    archive_files = glob.glob(archive_pattern)

    # Исправленная лямбда-сортировка по индексам архивов
    archive_files.sort(key=lambda x: [int(s) for s in x.split('.') if s.isdigit()][0], reverse=True)

    for gz_file in archive_files:
        print(f"🔓 Распаковываю и читаю архив: {gz_file}")
        try:
            with gzip.open(gz_file, 'rt') as f:
                # Читаем как строки, чтобы текстовый заголовок не ломал типы данных
                df_chunk = pd.read_csv(f, header=None, dtype=str)
                all_chunks.append(df_chunk)
        except Exception as e:
            print(f"❌ Ошибка чтения архива {gz_file}: {e}")

    if not all_chunks:
        print("❌ Данные для склейки не найдены!")
        return

    # 3. Объединяем все куски в одну большую матрицу
    df_monolith = pd.concat(all_chunks, ignore_index=True)

    # Присваиваем временные имена колонкам
    df_monolith.columns = ["timestamp"] + [f"col_{i}" for i in range(1, len(df_monolith.columns))]

    # 🔥 ЖЕСТКИЙ ФИЛЬТР: Выбрасываем строки, если в колонке timestamp случайно продублировался текст заголовка
    df_monolith = df_monolith[df_monolith["timestamp"] != "timestamp"]

    print("⚙️ Конвертирую временную сетку...")
    # Используем 'mixed', чтобы Pandas переварил любые микро-сдвиги форматов
    df_monolith["timestamp"] = pd.to_datetime(df_monolith["timestamp"], errors='coerce', format='mixed')

    # Удаляем строки, которые не удалось сконвертировать (на всякий случай)
    df_monolith = df_monolith.dropna(subset=["timestamp"])

    # Гарантируем идеальную хронологию и удаляем возможные нахлесты в точках слияния
    df_monolith = df_monolith.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # Возвращаем типам данных их исходный вид перед записью
    for col in df_monolith.columns:
        if col != "timestamp":
            df_monolith[col] = pd.to_numeric(df_monolith[col], errors='coerce')

    # ИСПРАВЛЕНИЕ: Назначаем реальные имена колонок очищенному датафрейму перед сохранением
    df_monolith.columns = headers_list

    # ИСПРАВЛЕНИЕ: Меняем header=False на header=True, чтобы записать строку с названиями колонок в начало файла
    df_monolith.to_csv(output_path, index=False, header=True)
    print(f"🎉 Сборка завершена! Монолитный файл сохранен: {output_path} ({len(df_monolith)} строк)")

if __name__ == "__main__":
    compile_massive_dataset()
