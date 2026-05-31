import numpy as np
import pandas as pd
import configparser


def make_impulse_time_machine():

    config = configparser.ConfigParser()
    config.read("config.ini")

    # Забираем макро-настройки разметки
    look_ahead = config.getint("LABELER", "look_ahead")
    # ПОРОГ ФИЛЬТРАЦИИ ШУМА (в долларах для BTC)
    noise_threshold = config.getfloat("LABELER", "noise_threshold")
    thinning_step = config.getint("LABELER", "data_thinning_step")
    csv_fileRow_data = config.get("LABELER", "csv_fileRow_data")

    print(f"📖 Читаем сырой файл {csv_fileRow_data}...")
    try:
        df = pd.read_csv(csv_fileRow_data)
    except FileNotFoundError:
        print("❌ Файл не найден.")
        return

    if len(df) < 50:
        print("❌ Мало данных.")
        return

    # ==========================================
    # 📡 МЯГКАЯ СИНХРОНИЗАЦИЯ СЕТКИ ВРЕМЕНИ
    # ==========================================
    print("⚙️ Выравниваю временную сетку без потери структуры...")

    # Форматируем время и гарантируем хронологический порядок строк (один раз)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Защита от NaN (если логгер записал пустые значения в момент микро-сбоя)
    # Цену и стаканы копируем из предыдущей известной секунды
    df["price"] = df["price"].ffill()

    imb_cols = ["imbalance_5", "imbalance_20", "imbalance_50"]
    df[imb_cols] = df[imb_cols].ffill()

    # Скорость и дельту заполняем нулями (если данных нет — значит активности не было)
    df["trade_speed_10s"] = df["trade_speed_10s"].fillna(0.0)
    df["market_delta_10s"] = df["market_delta_10s"].fillna(0.0)
    # ==========================================


    # ==========================================
    # 🧪 FEATURE ENGINEERING (КОНТЕКСТ)
    # ==========================================
    rolling_speed_mean = df["trade_speed_10s"].rolling(window=30, min_periods=5).mean()
    rolling_speed_std = df["trade_speed_10s"].rolling(window=30, min_periods=5).std()
    # Безопасный расчет Z-Score
    df["speed_zscore"] = ((df["trade_speed_10s"] - rolling_speed_mean) / rolling_speed_std)
    df["speed_zscore"] = df["speed_zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["delta_rolling_2m"] = df["market_delta_10s"].rolling(window=12, min_periods=3).sum().fillna(0)
    df["delta_rolling_5m"] = df["market_delta_10s"].rolling(window=30, min_periods=5).sum().fillna(0)
    df["imb_20_velocity"] = df["imbalance_20"] - df["imbalance_20"].shift(6)

    # 🚀 ДОБАВЛЯЕМ МАКРО-ФИЧИ ДЛЯ ЧАСОВОГО ТРЕНДА:
    # 1. Кумулятивная дельта за 30 минут (30 * 6 = 180 строк)
    df["delta_rolling_30m"] = df["market_delta_10s"].rolling(window=180, min_periods=30).sum().fillna(0)
    # 2. Кумулятивная дельта за 1 час (60 * 6 = 360 строк)
    df["delta_rolling_1h"] = df["market_delta_10s"].rolling(window=360, min_periods=60).sum().fillna(0)
    # 3. Скорость изменения цены за последние 15 минут (15 * 6 = 90 строк)
    df["price_velocity_15m"] = df["price"] - df["price"].shift(90)
    df["price_velocity_15m"] = df["price_velocity_15m"].fillna(0)

    # ==========================================
    # ⚡ ИМПУЛЬСНАЯ РАЗМЕТКА (3 КЛАССА)
    # ==========================================
    df["future_price"] = df["price"].shift(-look_ahead)

    # Считаем чистое изменение цены в долларах
    df["price_change"] = df["future_price"] - df["price"]


    # Правильная разметка для ML: 1 (Лонг), -1 (Шорт), 0 (Шум/Сидим в кэше)
    conditions = [
        (df["price_change"] > noise_threshold),
        (df["price_change"] < -noise_threshold),
    ]
    choices = [1, -1]

    df["label_next_price"] = np.select(conditions, choices, default=0)

    # 1. Очищаем строки с NaN по краям (где rolling / shift не посчитались и future_price = NaN)
    df_cleaned = df.dropna(
        subset=["future_price", "imb_20_velocity", "speed_zscore"]
    ).copy()

    # 2. Делаем разрежение по фиксированной сетке времени
    # Берем каждую 3-ю строку от реального потока времени для снижения автокорреляции
    df_filtered = df_cleaned.iloc[::thinning_step].reset_index(drop=True)

    # ВАЖНО: Мы НЕ удаляем класс 0 (шум). ИИ должен учиться понимать, когда в рынок лезть не надо.

    # 3. Удаляем лишние колонки (чтобы ИИ не подглядывал в будущее)
    df_filtered = df_filtered.drop(columns=["future_price", "price_change"])

    # Сохраняем новую качественную матрицу
    ready_file = "../../data/multidim_labeled_market_data.csv"
    df_filtered.to_csv(ready_file, index=False)

    print(f"🎉 Новая импульсная разметка завершена, произведено разрежение сетки!")
    print(f"🗑️ Было строк до фильтрации:                                        {len(df_cleaned)}")
    print(f"🎯 Итоговый размер датасета для ИИ (включая флэт-паттерны):         {len(df_filtered)}")
    print(f"📊 Баланс классов:\n{df_filtered['label_next_price'].value_counts().to_string()}")


if __name__ == "__main__":
    make_impulse_time_machine()
