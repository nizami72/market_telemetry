import os
import polars as pl
import configparser
# 🚨 ИМПОРТИРУЕМ НАШ НОВЫЙ МОДУЛЬ
from market_regime import detect_and_save_market_regime


def make_impulse_time_machine():
    # Находим абсолютный путь к config.ini относительно текущего скрипта
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.ini")

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    # Базовый путь к сырым данным
    csv_file_row_data = config.get("LABELER", "csv_filerow_data")
    look_ahead = config.getint("LABELER", "look_ahead")

    # ==========================================
    # 🧠 Update config file
    # ==========================================
    detect_and_save_market_regime(24)
    config.read(config_path, encoding="utf-8") # Перечитываем апдейты!

    # 🔧 Получаем динамические параметры
    noise_threshold = config.getfloat("LABELER", "noise_threshold")
    thinning_step = config.getint("LABELER", "data_thinning_step")
    # ==========================================

    print(f"📖 Читаем сырой файл {csv_file_row_data}...")
    try:
        # В Polars чтение CSV происходит молниеносно
        df = pl.read_csv(csv_file_row_data)
    except FileNotFoundError:
        print(f"❌ Файл не найден по пути: {csv_file_row_data}")
        return

    if df.height < 50:
        print("❌ Мало данных.")
        return

    # ==========================================
    # 📡 МЯГКАЯ СИНХРОНИЗАЦИЯ СЕТКИ ВРЕМЕНИ (UTC)
    # ==========================================
    print("⚙️ Выравниваю временную сетку в жестком формате UTC...")

    # В Polars операции делаются через .with_columns() и выражения
    df = (
        df.with_columns(
            # Жестко парсим строку в дату с таймзоной UTC
            pl.col("timestamp").str.to_datetime(time_zone="UTC")
        )
        .sort("timestamp")
        .with_columns([
            pl.col("price").forward_fill(),
            pl.col("imbalance_5").forward_fill(),
            pl.col("imbalance_20").forward_fill(),
            pl.col("imbalance_50").forward_fill(),
            pl.col("trade_speed_10s").fill_null(0.0),
            pl.col("market_delta_10s").fill_null(0.0),
        ])
    )

    # ==========================================
    # 🧪 FEATURE ENGINEERING (КОНТЕКСТ НА СКОРОСТИ RUST)
    # ==========================================
    print("⚙️ Генерирую расширенные фичи объемов и окон...")

    # Скользящие окна в Polars пишутся через rolling_mean/rolling_std
    df = df.with_columns([
        pl.col("trade_speed_10s").rolling_mean(window_size=30, min_periods=5).alias("speed_mean"),
        pl.col("trade_speed_10s").rolling_std(window_size=30, min_periods=5).alias("speed_std"),
    ]).with_columns([
        # Расчет z-score
        ((pl.col("trade_speed_10s") - pl.col("speed_mean")) / pl.col("speed_std"))
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("speed_zscore")
    ])

    # Основной пул фичей через мощный механизм выражений Polars (выполняется параллельно!)
    df = df.with_columns([
        pl.col("market_delta_10s").rolling_sum(12, min_periods=3).fill_null(0).alias("delta_rolling_2m"),
        pl.col("market_delta_10s").rolling_sum(30, min_periods=5).fill_null(0).alias("delta_rolling_5m"),
        (pl.col("imbalance_20") - pl.col("imbalance_20").shift(6)).fill_null(0).alias("imb_20_velocity"),

        pl.col("market_delta_10s").rolling_sum(180, min_periods=30).fill_null(0).alias("delta_rolling_30m"),
        pl.col("market_delta_10s").rolling_sum(360, min_periods=60).fill_null(0).alias("delta_rolling_1h"),
        (pl.col("price") - pl.col("price").shift(90)).fill_null(0).alias("price_velocity_15m"),
    ])

    # Высокочастотные фичи ускорения объемов
    df = df.with_columns([
        (pl.col("trade_speed_10s") / (pl.col("trade_speed_10s").rolling_mean(6, min_periods=1) + 1e-5)).alias("speed_ratio_1m"),
        (pl.col("trade_speed_10s") / (pl.col("trade_speed_10s").rolling_mean(30, min_periods=1) + 1e-5)).alias("speed_ratio_5m"),
        (pl.col("trade_speed_10s") / (pl.col("trade_speed_10s").rolling_mean(90, min_periods=1) + 1e-5)).alias("speed_ratio_15m"),

        pl.col("market_delta_10s").rolling_sum(6, min_periods=1).fill_null(0).alias("cum_delta_1m"),
        pl.col("market_delta_10s").rolling_sum(30, min_periods=1).fill_null(0).alias("cum_delta_5m"),
        pl.col("market_delta_10s").rolling_sum(90, min_periods=1).fill_null(0).alias("cum_delta_15m"),

        (pl.col("price") - pl.col("price").shift(30)).fill_null(0).alias("price_change_5m"),
        (pl.col("price") - pl.col("price").shift(360)).fill_null(0).alias("price_change_1h"),
    ])

    # Дропаем временные колонки средних, они больше не нужны
    df = df.drop(["speed_mean", "speed_std"])

    # ==========================================
    # ⚡ ДИНАМИЧЕСКАЯ ИМПУЛЬСНАЯ РАЗМЕТКА
    # ==========================================
    df = df.with_columns([
        pl.col("price").shift(-look_ahead).alias("future_price")
    ]).with_columns([
        (pl.col("future_price") - pl.col("price")).alias("price_change")
    ])

    # Аналог np.select в Polars — это конструкция pl.when().then().otherwise()
    df = df.with_columns(
        pl.when(pl.col("price_change") > noise_threshold).then(1)
        .when(pl.col("price_change") < -noise_threshold).then(-1)
        .otherwise(0)
        .alias("label_next_price")
    )

    # Запоминаем размер до очистки строк
    len_before_drop = df.height

    # Очищаем строки с нуллами в ключевых колонках (после shift)
    df_cleaned = df.drop_nulls(subset=["future_price", "imb_20_velocity", "speed_zscore"])

    # 🎯 ФИКС: Убираем пустые строки, которые вылезли в самом начале файла
    df_cleaned = df_cleaned.filter(pl.col("price").is_not_null() & pl.col("timestamp").is_not_null())

    # Разрежение сетки (аналог iloc[::thinning_step]) через метод gather_every
    df_filtered = df_cleaned.gather_every(thinning_step)

    # Удаляем ненужные для ИИ колонки
    df_filtered = df_filtered.drop(["future_price", "price_change"])

    ready_file = "../../data/multidim_labeled_market_data.csv"

    # 🎯 Polars запишет дату строго в формате ISO 8601: 2026-05-30T17:19:19+00:00
    df_filtered.write_csv(ready_file)

    print(f"🎉 Новая импульсная разметка на Polars завершена!")
    print(f"🗑️ Было строк до фильтрации:                                         {len_before_drop}")
    print(f"🎯 Итоговый размер датасета для ИИ (включая флэт-паттерны):         {df_filtered.height}")

    # Считаем баланс классов
    class_counts = df_filtered["label_next_price"].value_counts()
    print(f"📊 Баланс классов:\n{class_counts}")


if __name__ == "__main__":
    make_impulse_time_machine()
