import asyncio
import joblib
import numpy as np
import pandas as pd
import configparser
import collections
import ccxt.pro as ccxt
import os  # Убедись, что импорт os на месте

# Настройки API Bybit Testnet (Демо-аккаунт)
config = configparser.ConfigParser()

# 🎯 Вычисляем абсолютный путь к папке, где лежит САМ этот скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))
# Привязываем config.ini строго к этой же папке
config_path = os.path.join(current_dir, "config.ini")

print(f"📡 Загрузка конфигурации из: {config_path}")
config.read(config_path)

# Защитный ассерт, чтобы сразу поймать ошибку, если файла нет
if not config.has_section("API_KEYS"):
    raise FileNotFoundError(f"❌ Файл конфига не найден или пуст по пути: {config_path}")

API_KEY = config.get("API_KEYS", "bybit_api_key")
API_SECRET = config.get("API_KEYS", "bybit_api_secret")

print("🟢 Ключи успешно считаны из конфига!")

# Глобальные буферы данных из веб-сокетов
latest_order_book = None
trade_volume_buy_10s = 0.0
trade_volume_sell_10s = 0.0
trade_count_10s = 0

# Очередь для расчета скользящих макро-окон прямо в RAM (максимум за 1 час = 360 тиков)
history_buffer = collections.deque(maxlen=360)

async def order_book_listener(exchange, symbol):
    global latest_order_book
    while True:
        try:
            latest_order_book = await exchange.watch_order_book(symbol, limit=50)
        except Exception as e:
            print(f"❌ Ошибка WS стакана: {e}")
            await asyncio.sleep(2)

async def trades_listener(exchange, symbol):
    global trade_volume_buy_10s, trade_volume_sell_10s, trade_count_10s
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            for trade in trades:
                trade_count_10s += 1
                volume = trade["amount"]
                if trade["side"] == "buy":
                    trade_volume_buy_10s += volume
                else:
                    trade_volume_sell_10s += volume
        except Exception as e:
            print(f"❌ Ошибка WS ленты сделок: {e}")
            await asyncio.sleep(2)

async def execution_engine(exchange, symbol):
    global latest_order_book, trade_volume_buy_10s, trade_volume_sell_10s, trade_count_10s, config, current_dir, config_path
    # --- АВТОМАТИЧЕСКИЙ РАСЧЕТ ПУТИ К МОДЕЛИ ---
    # Берем имя файла из конфига или хардкодим, если оно фиксированное
    model_name = config.get("MARKET_DATA", "model_file_prod", fallback="lgbm_live_model.pkl")
    model_path = os.path.join(current_dir, model_name)

    print(f"🤖 Загружаю модель ИИ из: {model_path}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл модели {model_name} не найден по пути: {model_path}!")

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Файл модели lgbm_market_model.pkl не найден на сервере!")
        return

    # Запрашиваем баланс кошелька на Bybit перед стартом
    print("💰 Запрашиваю стартовый баланс на Bybit...")
    try:
        balance_info = await exchange.fetch_balance()
        current_balance = float(balance_info['total']['USDT'])
        print(f"💰 Боевой баланс подтвержден: {current_balance:.2f} USDT")
    except Exception as e:
        print(f"⚠️ Не удалось получить баланс по API ({e}). Ставлю дефолтные $10,000.")
        current_balance = 10000.0

    current_position = 0  # 0 = вне рынка, 1 = LONG, -1 = SHORT
    entry_price = 0.0
    pos_size_btc = 0.0

    print("🚀 Торговый движок синхронизирован с WebSocket-буфером. Начинаю рабочий цикл...")

    while True:
        await asyncio.sleep(10) # Шаг сетки — 10 секунд

        if latest_order_book is None or trade_count_10s == 0:
            continue

        # Читаем config.ini НА ЛЕТУ на каждом шаге (чтобы подхватывать Штиль/Шторм от market_regime.py)
        config = configparser.ConfigParser()
        config.read(config_path)
        try:
            threshold = config.getfloat("BACKTESTER", "threshold")
            tp_sl_size = config.getfloat("BACKTESTER", "tp_sl_size")
            risk_per_trade = config.getfloat("BACKTESTER", "risk_per_trade")
        except Exception as e:
            print(f"⚠️ Ошибка чтения config.ini ({e}). Применяю безопасные макро-параметры.")
            threshold, tp_sl_size, risk_per_trade = 0.42, 450.0, 0.01

        # 1. Расчет базовых тиковых параметров
        current_price = (latest_order_book["bids"][0][0] + latest_order_book["asks"][0][0]) / 2

        def calc_imbalance(depth):
            b_depth = sum([p * v for p, v in latest_order_book["bids"][:depth]])
            a_depth = sum([p * v for p, v in latest_order_book["asks"][:depth]])
            return b_depth / (b_depth + a_depth) if (b_depth + a_depth) > 0 else 0.5

        imb_5 = calc_imbalance(5)
        imb_20 = calc_imbalance(20)
        imb_50 = calc_imbalance(50)
        market_delta = trade_volume_buy_10s - trade_volume_sell_10s
        speed = trade_count_10s

        # Очищаем буфер ленты для следующих 10 секунд
        trade_volume_buy_10s = 0.0
        trade_volume_sell_10s = 0.0
        trade_count_10s = 0

        # Сохраняем текущий срез физики в оперативную память истории
        current_tick = {
            "price": current_price, "imbalance_5": imb_5, "imbalance_20": imb_20, "imbalance_50": imb_50,
            "market_delta_10s": market_delta, "trade_speed_10s": speed
        }
        history_buffer.append(current_tick)

        if len(history_buffer) < 30:
            print(f"⏳ Накапливаю RAM-буфер истории для расчета окон... ({len(history_buffer)}/30)")
            continue

        # 2. FEATURE ENGINEERING НА ЛЕТУ В ОПЕРАТИВНОЙ ПАМЯТИ
        # Превращаем буфер в легкий DataFrame для быстрого скользящего расчета признаков
        df_buf = pd.DataFrame(list(history_buffer))

        # Расчет Z-score скорости
        r_speed = df_buf["trade_speed_10s"].tail(30)
        s_mean = r_speed.mean()
        s_std = r_speed.std() if r_speed.std() > 0 else 1.0
        speed_zscore = (speed - s_mean) / s_std

        # Расчет оконных дельт и скоростей
        delta_rolling_2m = df_buf["market_delta_10s"].tail(12).sum()
        delta_rolling_5m = df_buf["market_delta_10s"].tail(30).sum()
        imb_20_velocity = imb_20 - df_buf["imbalance_20"].iloc[-7] if len(df_buf) >= 7 else 0.0

        delta_rolling_30m = df_buf["market_delta_10s"].tail(180).sum() if len(df_buf) >= 180 else df_buf["market_delta_10s"].sum()
        delta_rolling_1h = df_buf["market_delta_10s"].sum() # максимум 360 строк
        price_velocity_15m = current_price - df_buf["price"].iloc[-90] if len(df_buf) >= 90 else current_price - df_buf["price"].iloc[0]

        speed_ratio_1m = speed / (df_buf["trade_speed_10s"].tail(6).mean() + 1e-5)
        speed_ratio_5m = speed / (df_buf["trade_speed_10s"].tail(30).mean() + 1e-5)
        speed_ratio_15m = speed / (df_buf["trade_speed_10s"].tail(90).mean() + 1e-5)

        cum_delta_1m = df_buf["market_delta_10s"].tail(6).sum()
        cum_delta_5m = df_buf["market_delta_10s"].tail(30).sum()
        cum_delta_15m = df_buf["market_delta_10s"].tail(90).sum()

        price_change_5m = current_price - df_buf["price"].iloc[-30] if len(df_buf) >= 30 else 0.0
        price_change_1h = current_price - df_buf["price"].iloc[0]

        # Собираем вектор фичей ровно в том порядке, в котором учился LightGBM!
        X_live = np.array([[
            imb_5, imb_20, imb_50, market_delta, speed, speed_zscore,
            delta_rolling_2m, delta_rolling_5m, imb_20_velocity,
            delta_rolling_30m, delta_rolling_1h, price_velocity_15m,
            speed_ratio_1m, speed_ratio_5m, speed_ratio_15m,
            cum_delta_1m, cum_delta_5m, cum_delta_15m,
            price_change_5m, price_change_1h
        ]])

        # 3. ПОЛУЧЕНИЕ ВЕРОЯТНОСТЕЙ ИИ
        preds_proba = model.predict(X_live)
        proba_down = preds_proba[0, 0]
        proba_flat = preds_proba[0, 1]
        proba_up   = preds_proba[0, 2]

        # Логика Argmax + Порог (Зеркально нашему бэктестеру)
        signal = 0
        if proba_up > threshold:
            signal = 1
        elif proba_down > threshold:
            signal = -1

        # 4. ЛОГИКА ТОРГОВОГО КОНТУРА (Исполнение на Bybit)
        if current_position != 0:
            # Проверяем жесткие цели TP/SL в реальном времени
            is_closed = False

            if current_position == 1: # Мы в LONG
                change = current_price - entry_price
                if change >= tp_sl_size:
                    print(f"🟢 [LIVE TRIGGER] TAKE_PROFIT LONG достигнут на цене {current_price:.2f}!")
                    is_closed = True
                elif change <= -tp_sl_size:
                    print(f"🔴 [LIVE TRIGGER] STOP_LOSS LONG выбит на цене {current_price:.2f}!")
                    is_closed = True
                elif signal == -1:
                    print(f"🔄 [LIVE TRIGGER] Экстренный REVERSE CLOSE LONG по сигналу ИИ!")
                    is_closed = True

            elif current_position == -1: # Мы в SHORT
                change = entry_price - current_price
                if change >= tp_sl_size:
                    print(f"🟢 [LIVE TRIGGER] TAKE_PROFIT SHORT достигнут на цене {current_price:.2f}!")
                    is_closed = True
                elif change <= -tp_sl_size:
                    print(f"🔴 [LIVE TRIGGER] STOP_LOSS SHORT выбит на цене {current_price:.2f}!")
                    is_closed = True
                elif signal == 1:
                    print(f"🔄 [LIVE TRIGGER] Экстренный REVERSE CLOSE SHORT по сигналу ИИ!")
                    is_closed = True

            if is_closed:
                try:
                    # На бирже для закрытия позиции шлется противоположный ордер
                    side_to_close = "sell" if current_position == 1 else "buy"
                    print(f"📡 Отправляю ордер на ЗАКРЫТИЕ позиции: {side_to_close.upper()} объем {pos_size_btc:.4f} BTC")

                    # Боевой API-запрос на Bybit (Рыночный ордер закрытия)
                    order = await exchange.create_market_order(symbol, side_to_close, pos_size_btc, params={"category": "linear"})
                    print(f"✅ Ордер успешно исполнен биржей! ID: {order['id']}")

                    # Обновляем локальный баланс
                    balance_info = await exchange.fetch_balance()
                    current_balance = float(balance_info['total']['USDT'])
                    print(f"💰 Новый актуальный баланс: {current_balance:.2f} USDT")
                except Exception as e:
                    print(f"❌ Ошибка отправки ордера закрытия на Bybit: {e}")

                current_position = 0
                pos_size_btc = 0.0
                continue

        # Логика открытия позиции (только если мы вне рынка)
        if current_position == 0 and signal != 0:
            current_position = signal
            entry_price = current_price

            # РАСЧЕТ РИСК-МЕНЕДЖМЕНТА 1%
            cash_risk = current_balance * risk_per_trade
            pos_size_btc = round(cash_risk / tp_sl_size, 4) # Округляем до шага лота Bybit (4 знака)

            if pos_size_btc == 0:
                pos_size_btc = 0.0001 # минимальный лот BTC на деривативах

            try:
                side_to_open = "buy" if current_position == 1 else "sell"
                print(f"📡 [LIVE SIGNAL] Вхожу в {side_to_open.upper()}! Цена: {current_price:.2f} | Объем: {pos_size_btc} BTC | Вероятность: {max(proba_up, proba_down):.2f}")

                # Боевой API-запрос на Bybit (Рыночный ордер входа)
                order = await exchange.create_market_order(symbol, side_to_open, pos_size_btc, params={"category": "linear"})
                print(f"✅ Позиция успешно ОТКРЫТА! ID: {order['id']}")
            except Exception as e:
                print(f"❌ Ошибка отправки ордера открытия на Bybit: {e}")
                current_position = 0
                pos_size_btc = 0.0

        # Мониторинг в консоль Linux / journald
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Цена: {current_price:.1f} | Up/Flat/Down: {proba_up:.2f}/{proba_flat:.2f}/{proba_down:.2f} | Сигнал: {signal} | Поз: {current_position}")

async def main():
    # Инициализируем CCXT exchange для работы с Bybit деривативами (Линейные фьючерсы)
    exchange_params = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "linear"} # работаем с USDT деривативами
    }

    # ПЕРЕКЛЮЧАТЕЛЬ НА TESTNET (ДЕМО-АККАУНТ)
    exchange = ccxt.bybit(exchange_params)
    exchange.set_sandbox_mode(True) # ВКЛЮЧАЕТ РЕЖИМ ТЕСТНЕТА ФАНТИКОВ. Для реала — просто удалить эту строку.

    symbol = "BTC/USDT"

    # Запускаем фоновых воркеров сбора данных и ядро трейдера в общем Event Loop
    await asyncio.gather(
        order_book_listener(exchange, symbol),
        trades_listener(exchange, symbol),
        execution_engine(exchange, symbol)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Робот остановлен пользователем.")