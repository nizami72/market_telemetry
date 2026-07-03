import asyncio
import joblib
import numpy as np
import pandas as pd
import configparser
import collections
import ccxt.pro as ccxt
import os
import logging
import aiohttp  # Добавлен недостающий импорт для работы Telegram

# =====================================================================
# 1. СИСТЕМНОЕ ЛОГИРОВАНИЕ (ВЫДЕЛЕННЫЙ ЖУРНАЛ СДЕЛКИ)
# =====================================================================
logging.basicConfig(
    filename='paper_trading.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Вычисляем пути конфигурации
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.ini")

config = configparser.ConfigParser()
print(f"📡 Загрузка конфигурации из: {config_path}")
config.read(config_path)

# Глобальные буферы данных из веб-сокетов
latest_order_book = None
trade_volume_buy_10s = 0.0
trade_volume_sell_10s = 0.0
trade_count_10s = 0

# Очередь для расчета скользящих макро-окон прямо в RAM (максимум за 1 час = 360 тиков)
history_buffer = collections.deque(maxlen=360)


# =====================================================================
# 2. АСИНХРОННЫЙ ОТПРАВИТЕЛЬ TELEGRAM NOTIFICATION
# =====================================================================
async def send_paper_telegram_alert(text: str):
    """Легковесный асинхронный отправитель алертов"""
    local_config = configparser.ConfigParser()
    local_config.read("config.ini")

    try:
        token = local_config.get("TELEGRAM", "bot_token")
        chat_id = local_config.get("TELEGRAM", "chat_id")
    except Exception as e:
        print(f"⚠️ Ошибка чтения секции [TELEGRAM] в config.ini: {e}")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"⚠️ Ошибка отправки в TG: {response.status}")
    except Exception as e:
        print(f"❌ Сбой сети Telegram API: {e}")


# =====================================================================
# 3. ДВИЖОК ЭМУЛЯЦИИ ТОРГОВЛИ (PAPER TRADING ENGINE)
# =====================================================================
class PaperExecutor:
    def __init__(self, initial_balance=10000.0, taker_fee=0.0006):
        self.balance = initial_balance
        self.fee_rate = taker_fee  # Комиссия тейкера (0.06%)
        self.current_position = 0  # 0 = вне рынка, 1 = LONG, -1 = SHORT
        self.entry_price = 0.0
        self.pos_size_btc = 0.0

        print(f"\n[INIT] === ИНИЦИАЛИЗАЦИЯ PAPER TRADING ===")
        print(f"[INIT] Стартовый виртуальный баланс: ${self.balance:.2f} USDT")

        logging.info(f"=== РОБОТ ИНИЦИАЛИЗИРОВАН. СТАРТОВЫЙ БАЛАНС: ${self.balance:.2f} ===")

    def open_position(self, direction, price, tp_sl_size, risk_per_trade, confidence):
        """Эмуляция открытия позиции с жестким риск-менеджментом"""
        self.current_position = direction
        self.entry_price = price

        # Риск-менеджмент 1% от текущего виртуального баланса
        cash_risk = self.balance * risk_per_trade
        self.pos_size_btc = round(cash_risk / tp_sl_size, 4)
        if self.pos_size_btc == 0:
            self.pos_size_btc = 0.0001

        # Вычитаем комиссию за вход
        fee = (self.entry_price * self.pos_size_btc) * self.fee_rate
        self.balance -= fee

        pos_str = "LONG" if direction == 1 else "SHORT"
        tp_price = self.entry_price + tp_sl_size if direction == 1 else self.entry_price - tp_sl_size
        sl_price = self.entry_price - tp_sl_size if direction == 1 else self.entry_price + tp_sl_size

        msg = (f"🟢 [OPEN {pos_str}] | Цена входа: {self.entry_price:.2f} | Лот: {self.pos_size_btc} BTC | "
               f"Комиссия: ${fee:.4f} | Назначенные цели -> TP: ${tp_price:.1f} | SL: ${sl_price:.1f} | Баланс: ${self.balance:.2f}")
        print(msg)
        logging.info(msg)

        # Формируем красивый алерт для Telegram
        alert_msg = (
            f"📝 *PAPER TRADING: ВХОД В РЫНОК*\n"
            f"• Направление: `{pos_str}`\n"
            f"• Цена входа: `${price:,.2f}`\n"
            f"• Уверенность ИИ: `{confidence * 100:.1f}%`\n"
            f"• Виртуальный баланс: `${self.balance:,.2f} USDT`"
        )
        asyncio.create_task(send_paper_telegram_alert(alert_msg))

    def close_position(self, price, action_str, confidence):
        """Эмуляция закрытия позиции и фиксация прибыли/убытка"""
        pnl = 0.0
        if self.current_position == 1:  # LONG
            pnl = (price - self.entry_price) * self.pos_size_btc
        elif self.current_position == -1:  # SHORT
            pnl = (self.entry_price - price) * self.pos_size_btc

        # Вычитаем комиссию за выход
        fee = (price * self.pos_size_btc) * self.fee_rate
        net_pnl = pnl - fee
        self.balance += net_pnl

        msg = (f"🔄 [{action_str}] | Вход: {self.entry_price:.2f} -> Выход: {price:.2f} | "
               f"Чистый PnL: ${net_pnl:+.4f} | Комиссия: ${fee:.4f} | Новый Баланс: ${self.balance:.2f}")
        print(msg)
        logging.info(msg)

        status_icon = "🟢" if net_pnl > 0 else "🔴"
        alert_msg = (
            f"{status_icon} *PAPER TRADING: ЗАКРЫТИЕ ПОЗИЦИИ*\n"
            f"• Результат: `{action_str}`\n"
            f"• Цена выхода: `${price:,.2f}`\n"
            f"• Финансовый итог: `{net_pnl:+.2f} USDT`\n"
            f"• Уверенность ИИ в моменте: `{confidence * 100:.1f}%`\n"
            f"• Текущий баланс: `${self.balance:,.2f} USDT`"
        )
        asyncio.create_task(send_paper_telegram_alert(alert_msg))

        # Сброс параметров позиции после отправки алерта
        self.current_position = 0
        self.entry_price = 0.0
        self.pos_size_btc = 0.0


# Инициализация виртуального движка торговли
paper_engine = PaperExecutor(initial_balance=10000.0)


# =====================================================================
# 4. АСИНХРОННЫЕ СЛУШАТЕЛИ ВЕБ-СОКЕТОВ (CCXT.PRO)
# =====================================================================
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


# =====================================================================
# 5. ТОРГОВОЕ ЯДРО С РАСЧЕТОМ ФИЧЕЙ И МОДЕЛЬЮ
# =====================================================================
async def execution_engine(symbol):
    global latest_order_book, trade_volume_buy_10s, trade_volume_sell_10s, trade_count_10s, config, current_dir, config_path

    # Автоматический расчет пути к модели весов ИИ
    model_name = config.get("MARKET_DATA", "model_file_prod", fallback="../../../data/lgbm_live_model.pkl")
    model_path = os.path.join(current_dir, model_name)

    print(f"🤖 Загружаю модель ИИ из: {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл модели {model_name} не найден по пути: {model_path}!")

    model = joblib.load(model_path)
    print("🚀 Виртуальный торговый движок запущен и ждет наполнения RAM-буфера...")
    print(f"\n[INIT] === SENDING TELEGRAM NOTIFICATION ===")
    asyncio.create_task(send_paper_telegram_alert("The paper trading engine is ready!"))

    while True:
        await asyncio.sleep(10)  # Сетка — 10 секунд

        if latest_order_book is None or trade_count_10s == 0:
            continue

        # Читаем конфигурацию на лету (подхватываем Штиль/Шторм от market_regime.py)
        config = configparser.ConfigParser()
        config.read(config_path)
        try:
            threshold = config.getfloat("BACKTESTER", "threshold")
            tp_sl_size = config.getfloat("BACKTESTER", "tp_sl_size")
            risk_per_trade = config.getfloat("BACKTESTER", "risk_per_trade")
        except Exception as e:
            print(f"⚠️ Ошибка чтения config.ini ({e}). Безопасный режим.")
            threshold, tp_sl_size, risk_per_trade = 0.42, 450.0, 0.01

        # Ловим лучшие цены Аск/Бид и Спред
        current_ask = latest_order_book["asks"][0][0]
        current_bid = latest_order_book["bids"][0][0]
        current_price = (current_ask + current_bid) / 2

        # Расчет базовых тиковых дисбалансов
        def calc_imbalance(depth):
            b_depth = sum([p * v for p, v in latest_order_book["bids"][:depth]])
            a_depth = sum([p * v for p, v in latest_order_book["asks"][:depth]])
            return b_depth / (b_depth + a_depth) if (b_depth + a_depth) > 0 else 0.5

        imb_5 = calc_imbalance(5)
        imb_20 = calc_imbalance(20)
        imb_50 = calc_imbalance(50)
        market_delta = trade_volume_buy_10s - trade_volume_sell_10s
        speed = trade_count_10s

        # Очищаем буфер для следующего физического кванта времени
        trade_volume_buy_10s = 0.0
        trade_volume_sell_10s = 0.0
        trade_count_10s = 0

        # Сохраняем срез в оперативную память истории
        current_tick = {
            "price": current_price, "imbalance_5": imb_5, "imbalance_20": imb_20, "imbalance_50": imb_50,
            "market_delta_10s": market_delta, "trade_speed_10s": speed
        }
        history_buffer.append(current_tick)

        if len(history_buffer) < 30:
            print(f"⏳ Накапливаю RAM-буфер истории для расчета макро-фич... ({len(history_buffer)}/30)")
            continue

        # -----------------------------------------------------------------
        # FEATURE ENGINEERING НА ЛЕТУ В ОПЕРАТИВНОЙ ПАМЯТИ
        # -----------------------------------------------------------------
        df_buf = pd.DataFrame(list(history_buffer))

        r_speed = df_buf["trade_speed_10s"].tail(30)
        s_mean = r_speed.mean()
        s_std = r_speed.std() if r_speed.std() > 0 else 1.0
        speed_zscore = (speed - s_mean) / s_std

        delta_rolling_2m = df_buf["market_delta_10s"].tail(12).sum()
        delta_rolling_5m = df_buf["market_delta_10s"].tail(30).sum()
        imb_20_velocity = imb_20 - df_buf["imbalance_20"].iloc[-7] if len(df_buf) >= 7 else 0.0

        delta_rolling_30m = df_buf["market_delta_10s"].tail(180).sum() if len(df_buf) >= 180 else df_buf["market_delta_10s"].sum()
        delta_rolling_1h = df_buf["market_delta_10s"].sum()
        price_velocity_15m = current_price - df_buf["price"].iloc[-90] if len(df_buf) >= 90 else current_price - df_buf["price"].iloc[0]

        speed_ratio_1m = speed / (df_buf["trade_speed_10s"].tail(6).mean() + 1e-5)
        speed_ratio_5m = speed / (df_buf["trade_speed_10s"].tail(30).mean() + 1e-5)
        speed_ratio_15m = speed / (df_buf["trade_speed_10s"].tail(90).mean() + 1e-5)

        cum_delta_1m = df_buf["market_delta_10s"].tail(6).sum()
        cum_delta_5m = df_buf["market_delta_10s"].tail(30).sum()
        cum_delta_15m = df_buf["market_delta_10s"].tail(90).sum()

        price_change_5m = current_price - df_buf["price"].iloc[-30] if len(df_buf) >= 30 else 0.0
        price_change_1h = current_price - df_buf["price"].iloc[0]

        # Строго соблюдаем порядок признаков обучения LightGBM
        X_live = np.array([[
            imb_5, imb_20, imb_50, market_delta, speed, speed_zscore,
            delta_rolling_2m, delta_rolling_5m, imb_20_velocity,
            delta_rolling_30m, delta_rolling_1h, price_velocity_15m,
            speed_ratio_1m, speed_ratio_5m, speed_ratio_15m,
            cum_delta_1m, cum_delta_5m, cum_delta_15m,
            price_change_5m, price_change_1h
        ]])

        # ПОЛУЧЕНИЕ ПРОГНОЗА МОДЕЛИ
        preds_proba = model.predict(X_live)
        proba_down = preds_proba[0, 0]
        proba_flat = preds_proba[0, 1]
        proba_up   = preds_proba[0, 2]

        signal = 0
        current_confidence = 0.0
        if proba_up > threshold:
            signal = 1
            current_confidence = proba_up
        elif proba_down > threshold:
            signal = -1
            current_confidence = proba_down

        # -----------------------------------------------------------------
        # СКВОЗНАЯ СИМУЛЯЦИЯ КОНТУРА ВЫХОДОВ И ВХОДОВ (Зеркально живому боту)
        # -----------------------------------------------------------------
        if paper_engine.current_position != 0:
            is_closed = False
            action_label = ""
            execution_price = current_price

            if paper_engine.current_position == 1:  # Находимся в виртуальном LONG
                change = current_price - paper_engine.entry_price
                if change >= tp_sl_size:
                    execution_price = paper_engine.entry_price + tp_sl_size
                    action_label = "TAKE_PROFIT LONG"
                    is_closed = True
                elif change <= -tp_sl_size:
                    execution_price = paper_engine.entry_price - tp_sl_size
                    action_label = "STOP_LOSS LONG"
                    is_closed = True
                elif signal == -1:
                    execution_price = current_bid  # Закрываемся по цене спроса
                    action_label = "REVERSE CLOSE LONG"
                    is_closed = True

            elif paper_engine.current_position == -1:  # Находимся в виртуальном SHORT
                change = paper_engine.entry_price - current_price
                if change >= tp_sl_size:
                    execution_price = paper_engine.entry_price - tp_sl_size
                    action_label = "TAKE_PROFIT SHORT"
                    is_closed = True
                elif change <= -tp_sl_size:
                    execution_price = paper_engine.entry_price + tp_sl_size
                    action_label = "STOP_LOSS SHORT"
                    is_closed = True
                elif signal == 1:
                    execution_price = current_ask  # Выкупаем по цене предложения
                    action_label = "REVERSE CLOSE SHORT"
                    is_closed = True

            if is_closed:
                # Передаем уверенность ИИ на момент закрытия (для алертов)
                closing_conf = proba_down if paper_engine.current_position == 1 else proba_up
                paper_engine.close_position(execution_price, action_label, closing_conf)
                continue

        # Логика виртуального входа (только вне рынка)
        if paper_engine.current_position == 0 and signal != 0:
            # На демо-счете входим строго по Аску для LONG или по Биду для SHORT (честный спред!)
            entry_price_side = current_ask if signal == 1 else current_bid
            paper_engine.open_position(signal, entry_price_side, tp_sl_size, risk_per_trade, current_confidence)

        # Журналирование текущего состояния в консоль сервера
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Цена: {current_price:.1f} | "
              f"U/F/D: {proba_up:.2f}/{proba_flat:.2f}/{proba_down:.2f} | Сигнал: {signal} | Вирт_Поз: {paper_engine.current_position}")


# =====================================================================
# 6. ТОЧКА ВХОДА В ПРИЛОЖЕНИЕ
# =====================================================================
async def main():
    # Используем публичное WebSocket-подключение без ключей для эмуляции
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    symbol = "BTC/USDT"

    # Запускаем сбор и эмуляцию в едином Event Loop
    await asyncio.gather(
        order_book_listener(exchange, symbol),
        trades_listener(exchange, symbol),
        execution_engine(symbol)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Робот Paper Trader остановлен пользователем.")
