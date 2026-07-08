import os
import configparser
import pandas as pd
from telegram_alerts import send_telegram_alert_sync

def detect_and_save_market_regime(window_hours=24):
    """
    Читает путь к CSV из config.ini, анализирует волатильность (ATR) за 24 часа
    по файлу с заголовками (пропускает первую строку через skiprows=1)
    и жестко хардкодит актуальные фазовые настройки обратно в config.ini.
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
        # 🔬 ФИКС БАГА: ПРОПУСКАЕМ ЗАГОЛОВОК (skiprows=1) И ЧИТАЕМ ПО ИНДЕКСАМ
        # =====================================================================
        df = pd.read_csv(
            csv_path,
            header=None,
            skiprows=1,      # Пропускаем строку с текстом "timestamp,price", чтобы не ломать парсер
            usecols=[0, 1],
            names=["timestamp", "price"]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        df.set_index("timestamp", inplace=True)

        # =====================================================================
        # 🛡️ ГАРАНТИРУЕМ МОНОТОННОСТЬ И УБИРАЕМ НАХЛЕСТЫ ВРЕМЕНИ
        # =====================================================================
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        # =====================================================================

        # Берем временной срез за последние 24 часа
        last_time = df.index.max()
        start_time = last_time - pd.Timedelta(hours=window_hours)
        df_slice = df.loc[start_time:last_time]

        if df_slice.empty or len(df_slice) < 100:
            print("⚠️ Недостаточно свежих тиков для анализа. config.ini оставлен без изменений.")
            return

        # 3. Ресемплинг: агрегируем 10-секундные тики в 15-минутные свечи
        resampled = df_slice["price"].resample("15Min").ohlc()
        resampled.dropna(inplace=True)

        # Считаем средний размах (волатильность) одной свечи в долларах (ATR)
        avg_candle_range = (resampled["high"] - resampled["low"]).mean()

        # Читаем старый режим для проверки изменений (чтобы не спамить в TG)
        old_regime = config.get("BACKTESTER", "market_regime", fallback="НЕТ_ДАННЫХ")

        print(f"\n📊 РЕЗУЛЬТАТЫ МОНИТОРИНГА РЫНКА ЗА {window_hours} ЧАСА:")
        print(f"   · Средний размах 15-минутной свечи BTC: ${avg_candle_range:.2f}")

        # ДВУХУРОВНЕВАЯ АВТО-МАТЕМАТИКА РЕЖИМОВ:
        if avg_candle_range > 150.0:
            tp_sl_size = 300.0
            thinning_step = 90      # Сильное разрежение под макро-тренды
            threshold = 0.42        # Повышаем планку уверенности для ИИ в Шторм
            regime_name = "ШТОРМ (ВЫСОКАЯ ВОЛАТИЛЬНОСТЬ)"
        else:
            tp_sl_size = 300.0
            thinning_step = 45      # Плотный сбор под микро-паттерны
            threshold = 0.39        # Исправил опечатку: возвращаем флэтовый порог 0.39 из твоей архитектуры
            regime_name = "ШТИЛЬ (НИЗКАЯ ВОЛАТИЛЬНОСТЬ)"

        print(f"🔥 АКТИВИРОВАН РЕЖИМ: {regime_name}")

        # =====================================================================
        # 4. 💾 ЖЕСТКАЯ ЗАПИСЬ CONFIG.INI НА ЖЕСТКИЙ ДИСК VPS
        # =====================================================================
        if not config.has_section("LABELER"): config.add_section("LABELER")
        if not config.has_section("BACKTESTER"): config.add_section("BACKTESTER")

        config.set("LABELER", "data_thinning_step", str(thinning_step))

        config.set("BACKTESTER", "confidence_threshold", str(threshold))
        config.set("BACKTESTER", "tp_sl_size", str(tp_sl_size))
        config.set("BACKTESTER", "market_regime", regime_name)

        # Физически сохраняем изменения на диск
        with open(config_file, "w", encoding="utf-8") as f:
            config.write(f)

        print("💾 config.ini успешно синхронизирован на жестком диске!")

        # =====================================================================
        # 📢 УВЕДОМЛЕНИЕ В TELEGRAM (Только при реальной смене фазы рынка)
        # =====================================================================
        if old_regime != regime_name:
            alert_text = (
                f"🚨 *Контур MLOps: Смена фазы рынка!*\n\n"
                f"• *Новый regime:* `{regime_name}`\n"
                f"• *BTC ATR (24h):* `${avg_candle_range:.2f}`\n"
                f"• *Порог ИИ (Confidence):* `{threshold}`\n"
                f"• *Шаг разрежения матрицы:* `{thinning_step}` тиков."
            )
            send_telegram_alert_sync(alert_text)
            print("📢 Уведомление о смене режима отправлено в Telegram.")

    except Exception as e:
        error_msg = f"❌ Ошибка внутри модуля market_regime: {e}. Конфиг не изменен."
        print(error_msg)
        send_telegram_alert_sync(f"⚠️ *Критический сбой market_regime.py!*\nОшибка: `{e}`")

if __name__ == "__main__":
    detect_and_save_market_regime()
