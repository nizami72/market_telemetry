import os
import sys
import polars as pl
from market_regime import detect_and_save_market_regime


def make_impulse_time_machine(csv_file_row_data):
    # =====================================================================
    # 🧠 ДИНАМИЧЕСКИЙ АПДЕЙТ ВОЛАТИЛЬНОСТИ
    # =====================================================================
    # Разметчик сам инициирует перерасчет режима рынка ("Штиль"/"Шторм")
    # и обновляет config.ini на диске для контура обучения моделей.
    detect_and_save_market_regime(24)
    # =====================================================================

    # Динамически формируем имя итогового Feature Store файла
    base_name, _ = os.path.splitext(csv_file_row_data)
    csv_file_features_store = f"{base_name}_features.csv"


    print(f"📖 Читаем сырой файл {csv_file_row_data}...")
    try:
        df = pl.read_csv(csv_file_row_data)
    except FileNotFoundError:
        print(f"❌ Файл не найден по пути: {csv_file_row_data}")
        return

    if df.height < 50:
        print("❌ Недостаточно данных для расчета.")
        return

    # ==========================================
    # 📡 МЯГКАЯ СИНХРОНИЗАЦИЯ СЕТКИ ВРЕМЕНИ (UTC)
    # ==========================================
    print("⚙️ Выравниваю временную сетку в жестком формате UTC...")

    df = (
        df.with_columns(
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
    # 🧪 FEATURE ENGINEERING (СЛОЙ ВЫЧИСЛЕНИЙ)
    # ==========================================
    print("⚙️ Генерирую расширенные фичи объемов и окон на скорости Rust...")

    df = df.with_columns([
        pl.col("trade_speed_10s").rolling_mean(window_size=30, min_samples=5).alias("speed_mean"),
        pl.col("trade_speed_10s").rolling_std(window_size=30, min_samples=5).alias("speed_std"),
    ]).with_columns([
        ((pl.col("trade_speed_10s") - pl.col("speed_mean")) / pl.col("speed_std"))
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("speed_zscore")
    ])

    # Основной пул фичей через параллельный движок выражений Polars
    df = df.with_columns([
        pl.col("market_delta_10s").rolling_sum(12, min_samples=3).fill_null(0).alias("delta_rolling_2m"),
        pl.col("market_delta_10s").rolling_sum(30, min_samples=5).fill_null(0).alias("delta_rolling_5m"),
        (pl.col("imbalance_20") - pl.col("imbalance_20").shift(6)).fill_null(0).alias("imb_20_velocity"),

        pl.col("market_delta_10s").rolling_sum(180, min_samples=30).fill_null(0).alias("delta_rolling_30m"),
        pl.col("market_delta_10s").rolling_sum(360, min_samples=60).fill_null(0).alias("delta_rolling_1h"),
        (pl.col("price") - pl.col("price").shift(90)).fill_null(0).alias("price_velocity_15m"),
    ])

    # Высокочастотные фичи ускорения объемов
    df = df.with_columns([
        (pl.col("trade_speed_10s") / (pl.col("trade_speed_10s").rolling_mean(6, min_samples=1) + 1e-5)).alias("speed_ratio_1m"),
        (pl.col("trade_speed_10s") / (pl.col("trade_speed_10s").rolling_mean(30, min_samples=1) + 1e-5)).alias("speed_ratio_5m"),
        (pl.col("trade_speed_10s") / (pl.col("trade_speed_10s").rolling_mean(90, min_samples=1) + 1e-5)).alias("speed_ratio_15m"),

        pl.col("market_delta_10s").rolling_sum(6, min_samples=1).fill_null(0).alias("cum_delta_1m"),
        pl.col("market_delta_10s").rolling_sum(30, min_samples=1).fill_null(0).alias("cum_delta_5m"),
        pl.col("market_delta_10s").rolling_sum(90, min_samples=1).fill_null(0).alias("cum_delta_15m"),

        (pl.col("price") - pl.col("price").shift(30)).fill_null(0).alias("price_change_5m"),
        (pl.col("price") - pl.col("price").shift(360)).fill_null(0).alias("price_change_1h"),
    ])

    # Дропаем служебные колонки средних
    df = df.drop(["speed_mean", "speed_std"])

    # Запоминаем размер до удаления краевых нуллов
    len_before_drop = df.height

    # 🎯 Очищаем нуллы только на «прогревочном» старте фичей (первые скользящие окна)
    df_clean = df.drop_nulls(subset=["imb_20_velocity", "speed_zscore"])
    df_clean = df_clean.filter(pl.col("price").is_not_null() & pl.col("timestamp").is_not_null())

    # 🔥 ФИНАЛЬНОЕ СОХРАНЕНИЕ: Плотный, непрерывный Feature Store готов!
    df_clean.write_csv(csv_file_features_store)

    print(f"🎉 Сборка Feature Store на Polars успешно завершена!")
    print(f"💾 База признаков сохранена по пути: {csv_file_features_store}")
    print(f"📊 Исходных строк в логе:             {len_before_drop}")
    print(f"📊 Доступно чистых строк для ИИ:       {df_clean.height}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Ошибка запуска. Использование: python multidim_labeler.py <путь_к_сырому_файлу.csv>")
        sys.exit(1)

    target_csv = sys.argv[1]
    make_impulse_time_machine(target_csv)
