import numpy as np
import pandas as pd
import configparser
# 🚨 ИМПОРТИРУЕМ НАШ НОВЫЙ МОДУЛЬ
from market_regime import detect_market_regime


def make_impulse_time_machine():

    config = configparser.ConfigParser()
    config.read("config.ini")

    # Базовый путь к сырым данным
    csv_fileRow_data = config.get("LABELER", "csv_fileRow_data")
    look_ahead = config.getint("LABELER", "look_ahead")

    # ==========================================
    # 🧠 ДИНАМИЧЕСКИЙ АПГРЕЙД КОНФИГУРАЦИИ
    # ==========================================
    # Вызываем анализатор, который вернет настройки под текущий рынок
    dyn_noise_threshold, dyn_thinning_step, dyn_threshold = detect_market_regime(csv_fileRow_data)

    # Записываем динамические параметры в локальные переменные для разметчика
    noise_threshold = dyn_noise_threshold
    thinning_step = dyn_thinning_step

    # 🚨 ПЕРЕЗАПИСЫВАЕМ CONFIG.INI ДЛЯ ОБУЧЕНИЯ И БЭКТЕСТЕРА
    config.set("LABELER", "noise_threshold", str(dyn_noise_threshold))
    config.set("LABELER", "data_thinning_step", str(dyn_thinning_step))
    config.set("BACKTESTER", "threshold", str(dyn_threshold))
    config.set("BACKTESTER", "tp_sl_size", str(dyn_noise_threshold)) # Тейк равен макро-цели рынка

    with open("config.ini", "w") as configfile:
        config.write(configfile)

    print(f"💾 config.ini успешно обновлен на лету!")
    print(f"🎯 Рабочие параметры: Цель=${noise_threshold} | Шаг={thinning_step} | Порог ИИ={dyn_threshold}")
    # ===========================================

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

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["price"] = df["price"].ffill()

    imb_cols = ["imbalance_5", "imbalance_20", "imbalance_50"]
    df[imb_cols] = df[imb_cols].ffill()

    df["trade_speed_10s"] = df["trade_speed_10s"].fillna(0.0)
    df["market_delta_10s"] = df["market_delta_10s"].fillna(0.0)


    # ==========================================
    # 🧪 FEATURE ENGINEERING (КОНТЕКСТ)
    # ==========================================
    print("⚙️ Генерирую расширенные фичи объемов и окон...")

    rolling_speed_mean = df["trade_speed_10s"].rolling(window=30, min_periods=5).mean()
    rolling_speed_std = df["trade_speed_10s"].rolling(window=30, min_periods=5).std()

    df["speed_zscore"] = ((df["trade_speed_10s"] - rolling_speed_mean) / rolling_speed_std)
    df["speed_zscore"] = df["speed_zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["delta_rolling_2m"] = df["market_delta_10s"].rolling(window=12, min_periods=3).sum().fillna(0)
    df["delta_rolling_5m"] = df["market_delta_10s"].rolling(window=30, min_periods=5).sum().fillna(0)
    df["imb_20_velocity"] = df["imbalance_20"] - df["imbalance_20"].shift(6)

    df["delta_rolling_30m"] = df["market_delta_10s"].rolling(window=180, min_periods=30).sum().fillna(0)
    df["delta_rolling_1h"] = df["market_delta_10s"].rolling(window=360, min_periods=60).sum().fillna(0)
    df["price_velocity_15m"] = df["price"] - df["price"].shift(90)
    df["price_velocity_15m"] = df["price_velocity_15m"].fillna(0)

    # Высокочастотные фичи ускорения объемов
    df["speed_ratio_1m"] = df["trade_speed_10s"] / (df["trade_speed_10s"].rolling(window=6, min_periods=1).mean() + 1e-5)
    df["speed_ratio_5m"] = df["trade_speed_10s"] / (df["trade_speed_10s"].rolling(window=30, min_periods=1).mean() + 1e-5)
    df["speed_ratio_15m"] = df["trade_speed_10s"] / (df["trade_speed_10s"].rolling(window=90, min_periods=1).mean() + 1e-5)

    # Накопленная кумулятивная дельта
    df["cum_delta_1m"] = df["market_delta_10s"].rolling(window=6, min_periods=1).sum().fillna(0)
    df["cum_delta_5m"] = df["market_delta_10s"].rolling(window=30, min_periods=1).sum().fillna(0)
    df["cum_delta_15m"] = df["market_delta_10s"].rolling(window=90, min_periods=1).sum().fillna(0)

    # Скорости изменения тренда цены
    df["price_change_5m"] = (df["price"] - df["price"].shift(30)).fillna(0)
    df["price_change_1h"] = (df["price"] - df["price"].shift(360)).fillna(0)

    df = df.fillna(0.0)
    # ==========================================


    # ⚡ ДИНАМИЧЕСКАЯ ИМПУЛЬСНАЯ РАЗМЕТКА
    # ==========================================
    df["future_price"] = df["price"].shift(-look_ahead)
    df["price_change"] = df["future_price"] - df["price"]

    # Используем динамически определенный порог noise_threshold
    conditions = [
        (df["price_change"] > noise_threshold),
        (df["price_change"] < -noise_threshold),
    ]
    choices = [1, -1]

    df["label_next_price"] = np.select(conditions, choices, default=0)

    df_cleaned = df.dropna(
        subset=["future_price", "imb_20_velocity", "speed_zscore"]
    ).copy()

    # Разрежение сетки по динамическому шагу thinning_step
    df_filtered = df_cleaned.iloc[::thinning_step].reset_index(drop=True)

    df_filtered = df_filtered.drop(columns=["future_price", "price_change"])

    ready_file = "../../data/multidim_labeled_market_data.csv"
    df_filtered.to_csv(ready_file, index=False)

    print(f"🎉 Новая импульсная разметка завершена, произведено разрежение сетки!")
    print(f"🗑️ Было строк до фильтрации:                                        {len(df_cleaned)}")
    print(f"🎯 Итоговый размер датасета для ИИ (включая флэт-паттерны):         {len(df_filtered)}")
    print(f"📊 Баланс классов:\n{df_filtered['label_next_price'].value_counts().to_string()}")


if __name__ == "__main__":
    make_impulse_time_machine()
