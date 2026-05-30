import numpy as np
import pandas as pd


def make_impulse_time_machine():
    csv_file = "../../multidim_market_data.csv"

    print(f"📖 Читаем сырой файл {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("❌ Файл не найден.")
        return

    if len(df) < 50:
        print("❌ Мало данных.")
        return

    # Форматируем время
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ==========================================
    # 📡 СИНХРОНИЗАЦИЯ СЕТКИ ВРЕМЕНИ И ЛЕЧЕНИЕ ГЭПОВ
    # ==========================================
    print("⚙️ Выравниваю временную сетку (шаг 10с) и лечу сетевые дыры...")

    # Устанавливаем timestamp как индекс для работы с временными рядами
    df = df.set_index("timestamp")

    # Удаляем дубликаты индексов, если они случайно появились при сбоях
    df = df[~df.index.duplicated(keep='first')]

    # Создаем идеальную сетку с шагом 10 секунд
    # Все пропущенные отметки времени создадут новые пустые строки (NaN)
    df = df.resample("10s").asfreq()

    # Лечим пропуски:
    # 1. Цену (price) копируем вперед из последней известной точки (рынок стоял для нас)
    df["price"] = df["price"].ffill()

    # 2. Стаканы (imbalance) тоже берем из последнего известного состояния
    imb_cols = ["imbalance_5", "imbalance_20", "imbalance_50"]
    df[imb_cols] = df[imb_cols].ffill()

    # 3. Скорость и Дельту приравниваем к 0 (в моменты сбоя активности не было)
    df["trade_speed_10s"] = df["trade_speed_10s"].fillna(0.0)
    df["market_delta_10s"] = df["market_delta_10s"].fillna(0.0)

    # Возвращаем timestamp обратно в колонки и сбрасываем индекс
    df = df.reset_index()
    # ==========================================


    # ==========================================
    # 🧪 FEATURE ENGINEERING (КОНТЕКСТ)
    # ==========================================
    rolling_speed_mean = df["trade_speed_10s"].rolling(window=30, min_periods=5).mean()
    rolling_speed_std = df["trade_speed_10s"].rolling(window=30, min_periods=5).std()
   # Безопасный расчет Z-Score
    df["speed_zscore"] = ((df["trade_speed_10s"] - rolling_speed_mean) / rolling_speed_std)
    df["speed_zscore"] = df["speed_zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["delta_rolling_2m"] = (
        df["market_delta_10s"].rolling(window=12, min_periods=3).sum().fillna(0)
    )
    df["delta_rolling_5m"] = (
        df["market_delta_10s"].rolling(window=30, min_periods=5).sum().fillna(0)
    )
    df["imb_20_velocity"] = df["imbalance_20"] - df["imbalance_20"].shift(6)

    # ==========================================
    # ⚡ ИМПУЛЬСНАЯ РАЗМЕТКА И ФИЛЬТР ШУМА
    # ==========================================
    look_ahead = 90  # 3 минуты вперед
    df["future_price"] = df["price"].shift(-look_ahead)

    # Считаем чистое изменение цены в долларах
    df["price_change"] = df["future_price"] - df["price"]

    # ПОРОГ ФИЛЬТРАЦИИ ШУМА (в долларах для BTC)
    # Если за 3 минуты цена прошла меньше $15 — это рыночный шум
    noise_threshold = 20.0

    # Задаем условия: 1 - рост, 0 - падение, -1 - шум/флэт
    conditions = [
        (df["price_change"] > noise_threshold),
        (df["price_change"] < -noise_threshold),
    ]
    choices = [1, 0]

    df["label_next_price"] = np.select(conditions, choices, default=-1)

    # Очищаем строки с NaN (границы)
    df_cleaned = df.dropna(
        subset=["future_price", "imb_20_velocity", "speed_zscore"]
    ).copy()

    # 🚨 ИСКЛЮЧАЕМ ШУМ: выбрасываем все строки, где зафиксирован флэт (-1)
    df_filtered = df_cleaned[df_cleaned["label_next_price"] != -1].copy()

    # 🚨 РАЗРЕЖЕНИЕ ДАННЫХ (Data Thinning)
    # Чтобы убрать наложение окон, берем только каждую 3-ю строку из оставшихся (~раз в 30 сек)
    df_filtered = df_filtered.iloc[::3].reset_index(drop=True)

    # Удаляем лишние колонки
    df_filtered = df_filtered.drop(columns=["future_price", "price_change"])

    # Сохраняем новую качественную матрицу
    ready_file = "multidim_labeled_market_data.csv"
    df_filtered.to_csv(ready_file, index=False)

    print(f"🎉 Новая импульсная разметка завершена, выброшен шум и сделано разрежение!")
    print(f"🗑️ Было строк до фильтрации:                                        {len(df_cleaned)}")
    print(f"🎯 Осталось качественных импульсов для ИИ:                          {len(df_filtered)}"
    )


if __name__ == "__main__":
    make_impulse_time_machine()
