import os
import configparser
import pandas as pd
from telegram_alerts import send_telegram_alert_sync

def detect_and_save_market_regime(window_hours=24):
    """
    Читает путь к CSV из config.ini, анализирует волатильность (ATR) за 24 часа
    по файлу БЕЗ заголовков (использует 1 и 2 колонки) и жестко хардкодит
    актуальные фазовые настройки обратно в config.ini.
    """
    config_file = "config.ini"
    config = configparser.ConfigParser()

    # 1. Проверяем наличие конфигурационного файла
    if not os.path.exists(config_file):
        print(f"❌ Критическая ошибка: Файл конфигурации {config_file} не найден!")
        return

    try:
        # Читаем конфиг, чтобы узнать, где лежит сырой CSV логгера
        config.read(config_file)
        if not config.has_option("LABELER", "csv_filerow_data"):
            print("❌ Ошибка: В конфиге не задан параметр [LABELER] -> csv_filerow_data")
            return

        csv_path = config.get("LABELER", "csv_filerow_data")
        print(f"🧐 Конфиг подгружен. Анализирую фазу рынка по файлу: {csv_path}...")

        if not os.path.exists(csv_path):
            print(f"⚠️ Указанный CSV-файл {csv_path} еще не создан логгером.")
            return

        # =====================================================================
        # 🔬 МАНЕВР PANDAS: ЧТЕНИЕ БЕЗ ЗАГОЛОВКОВ ПО ИНДЕКСАМ КОЛОНОК
        # =====================================================================
        # header=None говорит, что первой строки с именами нет
        # usecols=[0, 1] загружает только 1-ю (timestamp) и 2-ю (price) колонки
        df = pd.read_csv(csv_path, header=None, usecols=[0, 1], names=["timestamp", "price"])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Берем временной срез за последние 24 часа
        last_time = df.index.max()
        start_time = last_time - pd.Timedelta(hours=window_hours)
        df_slice = df.loc[start_time:last_time]

        if df_slice.empty or len(df_slice) < 100:
            print("⚠️ Недостаточно свежих тиков для анализа. config.ini оставлен без изменений.")
            return

        # 2. Магия ресемплинга: агрегируем 10-секундные тики в 15-минутные свечи
        resampled = df_slice["price"].resample("15Min").ohlc()

        # Считаем средний размах (волатильность) одной свечи в долларах (ATR)
        avg_candle_range = (resampled["high"] - resampled["low"]).mean()

        # Читаем старый режим для проверки изменений (чтобы не спамить в TG одинаковыми сообщениями)
        old_regime = config.get("BACKTESTER", "market_regime", fallback="НЕТ_ДАННЫХ")

        print(f"\n📊 РЕЗУЛЬТАТЫ МОНИТОРИНГА РЫНКА ЗА {window_hours} ЧАСА:")
        print(f"   · Средний размах 15-минутной свечи BTC: ${avg_candle_range:.2f}")

        # ДВУХУРОВНЕВАЯ АВТО-МАТЕМАТИКА РЕЖИМОВ:
        if avg_candle_range > 150.0:
            noise_threshold = 450.0
            thinning_step = 90      # Разряжаем сильнее под макро-тренды
            threshold = 0.42        # Повышаем планку уверенности для ИИ (3 класса)
            regime_name = "ШТОРМ (ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ)"
        else:
            noise_threshold = 300.0
            thinning_step = 45      # Берем плотнее под микро-паттерны
            threshold = 0.39        # Чуть снижаем порог для флэта
            regime_name = "ШТИЛЬ (НИЗКАЯ ВОЛАТИЛЬНОСТЬ)"

        print(f"🔥 АКТИВИРОВАН РЕЖИМ: {regime_name}")

        # =====================================================================
        # 💾 ЖЕСТКАЯ ПЕРЕЗАПИСЬ CONFIG.INI НА ДИСКЕ VPS
        # =====================================================================
        if not config.has_section("LABELER"): config.add_section("LABELER")
        if not config.has_section("BACKTESTER"): config.add_section("BACKTESTER")

        config.set("LABELER", "noise_threshold", str(noise_threshold))
        config.set("LABELER", "data_thinning_step", str(thinning_step))

        config.set("BACKTESTER", "confidence_threshold", str(threshold))
        config.set("BACKTESTER", "tp_sl_size", str(noise_threshold)) # Синхронизируем тейки/стопы робота
        config.set("BACKTESTER", "market_regime", regime_name)

        with open(config_file, "w", encoding="utf-8") as f:
            config.write(f)

        print("💾 config.ini успешно синхронизирован на жестком диске!")
        # =====================================================================

        # =====================================================================
        # 📢 УВЕДОМЛЕНИЕ В TELEGRAM (Только при реальной смене фазы рынка)
        # =====================================================================
        if old_regime != regime_name:
            tg_message = (
                f"🔄 *MLOps Контур: Смена режима рынка!*\n\n"
                f"📊 *ATR 24h:* ${avg_candle_range:.2f}\n"
                f"⚙️ *Новая фаза:* `{regime_name}`\n"
                f"🎯 *Параметры ИИ:* Цель=${noise_threshold} | Порог={threshold}"
            )
            send_telegram_alert_sync(tg_message)
            print("📢 Уведомление о смене режима отправлено в Telegram.")
        # =====================================================================

    except Exception as e:
        print(f"❌ Ошибка внутри модуля market_regime: {e}. Конфиг не изменен.")

if __name__ == "__main__":
    detect_and_save_market_regime()
