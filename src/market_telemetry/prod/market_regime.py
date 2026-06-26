import pandas as pd
import numpy as np

def detect_market_regime(csv_path, window_hours=24):
    """
    Анализирует волатильность (размах цены) за последние сутки
    и возвращает динамические настройки для разметчика и ИИ.
    """
    try:
        print(f"🧐 Анализирую фазу рынка по файлу {csv_path}...")

        # Читаем только нужные колонки, чтобы сберечь RAM Hetzner-сервера
        df = pd.read_csv(csv_path, usecols=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Берем временной срез за последние 24 часа
        last_time = df.index.max()
        start_time = last_time - pd.Timedelta(hours=window_hours)
        df_slice = df.loc[start_time:last_time]

        if df_slice.empty or len(df_slice) < 100:
            print("⚠️ Недостаточно свежих данных для анализа волатильности. Режим: Дефолтный флэт.")
            return 300.0, 45, 0.39

        # Агрегируем 10-секундные тики в 15-минутные свечи OHLC
        resampled = df_slice["price"].resample("15Min").ohlc()

        # Считаем средний размах (волатильность) одной свечи в долларах
        avg_candle_range = (resampled["high"] - resampled["low"]).mean()

        print(f"\n📊 РЕЗУЛЬТАТЫ МОНИТОРИНГА РЫНКА ЗА {window_hours} ЧАСА:")
        print(f"   · Средний размах 15-минутной свечи BTC: ${avg_candle_range:.2f}")

        # ДВУХУРОВНЕВАЯ АВТО-МАТЕМАТИКА КОНФИГА:
        if avg_candle_range > 150.0:
            # Рынок живой, волатильный, трендовый
            noise_threshold = 450.0
            thinning_step = 90      # разряжаем сильнее, чтобы не ловить автокорреляцию на трендах
            threshold = 0.42        # повышаем планку уверенности для ИИ
            print("   🔥 РЕЖИМ: ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ. Включаю макро-цели ($450).")
        else:
            # Рынок тухлый, зажатый в боковик
            noise_threshold = 300.0
            thinning_step = 45      # берем данные плотнее, чтобы ИИ нашел микро-закономерности
            threshold = 0.39        # чуть снижаем порог, так как вероятности будут зажаты
            print("   💤 РЕЖИМ: НИЗКАЯ ВОЛАТИЛЬНОСТЬ (ФЛЭТ). Включаю микро-цели ($300).")

        return noise_threshold, thinning_step, threshold

    except Exception as e:
        print(f"❌ Ошибка внутри market_regime: {e}. Применяю безопасные флэт-настройки.")
        return 300.0, 45, 0.39