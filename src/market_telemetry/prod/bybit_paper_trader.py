import asyncio
import joblib
import numpy as np
import pandas as pd
import configparser
import collections
import ccxt.pro as ccxt
import os
import logging
from telegram_alerts import send_telegram_alert_async

# =====================================================================
# 1. СИСТЕМНОЕ ЛОГИРОВАНИЕ (ВЫДЕЛЕННЫЙ ЖУРНАЛ СДЕЛКИ)
# =====================================================================
logging.basicConfig(
    filename='paper_trading.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

# Плотная сетка 10-секундных тиков. Максимум за 1 час = 360 тиков (для price_velocity_15m нужно минимум 90)
history_buffer = collections.deque(maxlen=360)

# =====================================================================
# 3. ДВИЖОК ЭМУЛЯЦИИ ТОРГОВЛИ (PAPER TRADING ENGINE С ПОДДЕРЖКОЙ MUTEX)
# =====================================================================
class PaperExecutor:
    """
    A class to emulate trading operations (paper trading).
    Handles balance management, position sizing, and fee calculations.
    """
    def __init__(self, initial_balance=10000.0, taker_fee=0.0006):
        """
        Initializes the paper trading engine.

        :param initial_balance: Starting virtual balance in USDT.
        :param taker_fee: Taker fee rate (default is Bybit's 0.06%).
        """
        self.balance = initial_balance
        self.fee_rate = taker_fee  # Комиссия тейкера (0.06%) Bybit
        self.current_position = 0  # 0 = вне рынка, 1 = LONG, -1 = SHORT
        self.entry_price = 0.0
        self.pos_size_btc = 0.0

        # Замороженные RAM-уровни текущей сделки (для защиты от внутридневного дрейфа config.ini)
        self.active_tp_sl = 0.0
        self.active_agent = None

        print(f"\n[INIT] === ИНИЦИАЛИЗАЦИЯ МУЛЬТИОКАННОГО PAPER TRADING ===")
        print(f"[INIT] Стартовый виртуальный баланс: ${self.balance:.2f} USDT")
        logging.info(f"=== РОБОТ ИНИЦИАЛИЗИРОВАН. СТАРТОВЫЙ БАЛАНС: ${self.balance:.2f} ===")

    def open_position(self, direction, price, tp_sl_size, risk_per_trade, confidence, agent_name):
        """
        Simulates opening a position with fixed parameters.

        :param direction: 1 for LONG, -1 for SHORT.
        :param price: Entry price.
        :param tp_sl_size: Size of Take Profit and Stop Loss in USDT.
        :param risk_per_trade: Percentage of balance to risk per trade.
        :param confidence: Model's confidence level for the signal.
        :param agent_name: Name of the agent that triggered the signal.
        """
        self.current_position = direction
        self.entry_price = price
        self.active_tp_sl = tp_sl_size
        self.active_agent = agent_name

        # Динамический Position Sizing: строго 1% риска от капитала на сделку
        cash_risk = self.balance * risk_per_trade
        self.pos_size_btc = round(cash_risk / tp_sl_size, 4)
        if self.pos_size_btc == 0:
            self.pos_size_btc = 0.0001

        # Списание комиссии за Taker-вход
        fee = (self.entry_price * self.pos_size_btc) * self.fee_rate
        self.balance -= fee

        pos_str = "LONG" if direction == 1 else "SHORT"
        tp_price = self.entry_price + tp_sl_size if direction == 1 else self.entry_price - tp_sl_size
        sl_price = self.entry_price - tp_sl_size if direction == 1 else self.entry_price + tp_sl_size

        msg = (f"🟢 [MUTEX LOCKED BY {agent_name}] | Вход {pos_str}: {self.entry_price:.2f} | Лот: {self.pos_size_btc} BTC | "
               f"Комиссия: ${fee:.4f} | Цели -> TP: ${tp_price:.1f} | SL: ${sl_price:.1f} | Баланс: ${self.balance:.2f}")
        print(msg)
        logging.info(msg)

        alert_msg = (
            f"🎯 *PAPER TRADING: ВХОД В РЫНОК*\n"
            f"• Агент: `{agent_name}`\n"
            f"• Направление: `{pos_str}`\n"
            f"• Цена входа: `${price:,.2f}`\n"
            f"• Уверенность ИИ: `{confidence * 100:.1f}%`\n"
            f"• Риск-параметры (TP/SL): `±{tp_sl_size} USDT`\n"
            f"• Виртуальный баланс: `${self.balance:,.2f} USDT`"
        )
        asyncio.create_task(send_telegram_alert_async(alert_msg))

    def close_position(self, price, action_str, confidence):
        """
        Simulates closing a position, calculates PnL, and resets the state.

        :param price: Exit price.
        :param action_str: Description of the exit action (e.g., "TAKE_PROFIT").
        :param confidence: Model's confidence at the time of closing.
        """
        pnl = 0.0
        if self.current_position == 1:  # LONG
            pnl = (price - self.entry_price) * self.pos_size_btc
        elif self.current_position == -1:  # SHORT
            pnl = (self.entry_price - price) * self.pos_size_btc

        # Списание комиссии за Taker-выход
        fee = (price * self.pos_size_btc) * self.fee_rate
        net_pnl = pnl - fee
        self.balance += net_pnl

        msg = (f"🔓 [MUTEX UNLOCKED] | Агент: {self.active_agent} -> {action_str} | Вход: {self.entry_price:.2f} -> Выход: {price:.2f} | "
               f"Чистый PnL: ${net_pnl:+.4f} | Комиссия: ${fee:.4f} | Новый Баланс: ${self.balance:.2f}")
        print(msg)
        logging.info(msg)

        status_icon = "🟢" if net_pnl > 0 else "🔴"
        alert_msg = (
            f"{status_icon} *PAPER TRADING: ЗАКРЫТИЕ ПОЗИЦИИ*\n"
            f"• Агент: `{self.active_agent}`\n"
            f"• Исход: `{action_str}`\n"
            f"• Цена выхода: `${price:,.2f}`\n"
            f"• Финансовый итог: `{net_pnl:+.2f} USDT`\n"
            f"• Текущий баланс: `${self.balance:,.2f} USDT`"
        )
        asyncio.create_task(send_telegram_alert_async(alert_msg))

        # Полное освобождение Mutex-ресурсов для каскада
        self.current_position = 0
        self.entry_price = 0.0
        self.pos_size_btc = 0.0
        self.active_tp_sl = 0.0
        self.active_agent = None


paper_engine = PaperExecutor(initial_balance=10000.0)


# =====================================================================
# 4. АСИНХРОННЫЕ СЛУШАТЕЛИ ВЕБ-СОКЕТОВ (CCXT.PRO)
# =====================================================================
async def order_book_listener(exchange, symbol):
    """
    Asynchronous task to continuously watch the order book via WebSocket.

    :param exchange: CCXT exchange instance.
    :param symbol: Trading symbol (e.g., "BTC/USDT").
    """
    global latest_order_book
    while True:
        try:
            latest_order_book = await exchange.watch_order_book(symbol, limit=50)
        except Exception as e:
            print(f"❌ Ошибка WS стакана: {e}")
            await asyncio.sleep(2)

async def trades_listener(exchange, symbol):
    """
    Asynchronous task to continuously watch public trades via WebSocket.
    Updates global volume and trade count counters.

    :param exchange: CCXT exchange instance.
    :param symbol: Trading symbol (e.g., "BTC/USDT").
    """
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
# 5. ПАРАЛЛЕЛЬНЫЙ PREDICT STREAM И ТОРГОВОЕ ЯДРО КАСКАДА С MUTEX
# =====================================================================
async def execution_engine(symbol, cascade_models):
    """
    Main trading logic loop. Performs feature engineering, monitors active positions,
    and generates entry signals using a cascade of models.

    :param symbol: Trading symbol.
    :param cascade_models: Dictionary of loaded LightGBM models.
    """
    global latest_order_book, trade_volume_buy_10s, trade_volume_sell_10s, trade_count_10s, config, config_path

    # Читаем конфигурацию агентов из config.ini
    AGENTS_METADATA = {}
    try:
        agents_str = config.get("AGENTS", "list", fallback="M15,M30,M45,M60")
        agents_list = [a.strip() for a in agents_str.split(",")]
        for agent in agents_list:
            target = config.getfloat("AGENTS", f"{agent}_target_usdt", fallback=450.0)
            AGENTS_METADATA[agent] = {"target_usdt": target}
    except Exception as e:
        print(f"⚠️ Ошибка чтения AGENTS_METADATA из config.ini ({e}). Использую хардкод.", flush=True)
        AGENTS_METADATA = {
            "M15": {"target_usdt": 250.0},
            "M30": {"target_usdt": 450.0},
            "M45": {"target_usdt": 600.0},
            "M60": {"target_usdt": 800.0}
        }

    print("🚀 Параллельный Predict Stream запущен. Начинаю слушать рынок...", flush=True)
    asyncio.create_task(send_telegram_alert_async("⚡ Cascade Multi-Horizon Paper Trading Engine is ready!"))

    # Читаем глобальные настройки риска из config.ini
    try:
        confidence_threshold = config.getfloat("BACKTESTER", "confidence_threshold", fallback=0.55)
        risk_per_trade = config.getfloat("BACKTESTER", "risk_per_trade", fallback=0.01)
    except Exception as e:
        print(f"⚠️ Ошибка чтения config.ini ({e}). Безопасный дефолт.", flush=True)
        confidence_threshold, risk_per_trade = 0.55, 0.01

    while True:
        await asyncio.sleep(10)  # Строгий шаг контура — 10 секунд

        if latest_order_book is None:
            print("⏳ Ожидаю первое обновление стакана (WebSocket)...", flush=True)
            continue

        if trade_count_10s == 0:
            # Если сделок за 10 секунд не было, просто пропускаем тик, чтобы не делить на ноль
            continue

        # Фиксация цен краев стакана (для честной симуляции спреда) и Mid-Price (для признаков)
        current_ask = latest_order_book["asks"][0][0]
        current_bid = latest_order_book["bids"][0][0]
        current_price = (current_ask + current_bid) / 2

        # Расчет мгновенных характеристик стакана и ленты
        def calc_imbalance(depth):
            b_depth = sum([p * v for p, v in latest_order_book["bids"][:depth]])
            a_depth = sum([p * v for p, v in latest_order_book["asks"][:depth]])
            return b_depth / (b_depth + a_depth) if (b_depth + a_depth) > 0 else 0.5

        imb_5 = calc_imbalance(5)
        imb_20 = calc_imbalance(20)
        imb_50 = calc_imbalance(depth=50)
        market_delta = trade_volume_buy_10s - trade_volume_sell_10s
        speed = trade_count_10s

        # Мгновенный сброс накопителей под следующий 10-секундный квант
        trade_volume_buy_10s = 0.0
        trade_volume_sell_10s = 0.0
        trade_count_10s = 0

        # Аппендим срез в RAM-буфер истории
        current_tick = {
            "price": current_price, "imbalance_5": imb_5, "imbalance_20": imb_20, "imbalance_50": imb_50,
            "market_delta_10s": market_delta, "trade_speed_10s": speed
        }
        history_buffer.append(current_tick)

        # Минимальный порог плотности для расчета price_velocity_15m (90 тиков * 10с = 15 минут)
        if len(history_buffer) < 90:
            print(f"⏳ Накапливаю RAM-буфер истории макро-окон... ({len(history_buffer)}/90)", flush=True)
            continue

        # COMPUTE FEATURE ENGINEERING
        df_buf = pd.DataFrame(list(history_buffer))

        r_speed = df_buf["trade_speed_10s"].tail(30)
        s_mean = r_speed.mean()
        s_std = r_speed.std() if r_speed.std() > 0 else 1.0
        speed_zscore = (speed - s_mean) / s_std

        delta_rolling_2m = df_buf["market_delta_10s"].tail(12).sum()
        delta_rolling_5m = df_buf["market_delta_10s"].tail(30).sum()
        imb_20_velocity = imb_20 - df_buf["imbalance_20"].iloc[-7]

        delta_rolling_30m = df_buf["market_delta_10s"].tail(180).sum()
        delta_rolling_1h = df_buf["market_delta_10s"].sum()
        price_velocity_15m = current_price - df_buf["price"].iloc[-90]

        X_live = np.array([[
            imb_5, imb_20, imb_50, market_delta, speed, speed_zscore,
            delta_rolling_2m, delta_rolling_5m, imb_20_velocity,
            delta_rolling_30m, delta_rolling_1h, price_velocity_15m
        ]])

        # КОНТУР №1: ТЕКУЩЕЕ СОСТОЯНИЕ И МОНИТОРИНГ УДЕРЖАНИЯ ОРДЕРА
        if paper_engine.current_position != 0:
            is_closed = False
            action_label = ""
            execution_price = current_price
            frozen_targets = paper_engine.active_tp_sl

            if paper_engine.current_position == 1:
                change = current_price - paper_engine.entry_price
                if change >= frozen_targets:
                    execution_price = paper_engine.entry_price + frozen_targets
                    action_label = "TAKE_PROFIT LONG"
                    is_closed = True
                elif change <= -frozen_targets:
                    execution_price = paper_engine.entry_price - frozen_targets
                    action_label = "STOP_LOSS LONG"
                    is_closed = True

            elif paper_engine.current_position == -1:
                change = paper_engine.entry_price - current_price
                if change >= frozen_targets:
                    execution_price = paper_engine.entry_price - frozen_targets
                    action_label = "TAKE_PROFIT SHORT"
                    is_closed = True
                elif change <= -frozen_targets:
                    execution_price = paper_engine.entry_price + frozen_targets
                    action_label = "STOP_LOSS SHORT"
                    is_closed = True

            if is_closed:
                active_model = cascade_models[paper_engine.active_agent]
                exit_proba = active_model.predict(X_live)
                closing_conf = exit_proba[0, 1] if paper_engine.current_position == -1 else exit_proba[0, 2]
                paper_engine.close_position(execution_price, action_label, closing_conf)

            continue

        # КОНТУР №2: ПАРАЛЛЕЛЬНЫЙ PREDICT STREAM И ДИНАМИЧЕСКИЙ MUTEX ВХОД
        proba_logs = []
        for agent_name, model in cascade_models.items():
            preds_proba = model.predict(X_live)
            proba_down = preds_proba[0, 1]
            proba_up   = preds_proba[0, 2]
            proba_stay = preds_proba[0, 0]

            proba_logs.append(f"{agent_name}: [L:{proba_up:.2f} | S:{proba_down:.2f} | N:{proba_stay:.2f}]")

            agent_signal = 0
            confidence = 0.0
            if proba_up > confidence_threshold:
                agent_signal = 1
                confidence = proba_up
            elif proba_down > confidence_threshold:
                agent_signal = -1
                confidence = proba_down

            if agent_signal != 0:
                target_usdt = AGENTS_METADATA[agent_name]["target_usdt"]
                entry_price_side = current_ask if agent_signal == 1 else current_bid
                paper_engine.open_position(
                    direction=agent_signal, price=entry_price_side, tp_sl_size=target_usdt,
                    risk_per_trade=risk_per_trade, confidence=confidence, agent_name=agent_name
                )
                break

        # Pretty log of probabilities
        log_line = " | ".join(proba_logs)
        print(f"📊 Probabilities: {log_line}", flush=True)

        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] BTC: {current_price:.1f} | Buf: {len(history_buffer)} | Active Pos: {paper_engine.current_position}", flush=True)


# =====================================================================
# 6. ТОЧКА ВХОДА В ПРИЛОЖЕНИЕ (EVENT LOOP)
# =====================================================================
async def main():
    """
    Entry point for the Paper Trader. Initializes models, setup the exchange,
    and starts asynchronous listeners and the execution engine.
    """
    # Синхронная предобработка: проверяем веса каскада до запуска сокетов
    try:
        agents_str = config.get("AGENTS", "list", fallback="M15,M30,M45,M60")
        AGENTS_LIST = [a.strip() for a in agents_str.split(",")]
    except Exception as e:
        print(f"⚠️ Ошибка чтения AGENTS_LIST из config.ini ({e}). Использую дефолт.", flush=True)
        AGENTS_LIST = ["M15", "M30", "M45", "M60"]

    cascade_models = {}

    try:
        models_path = config.get("PATH", "models", fallback="../../../models")
    except Exception as e:
        print(f"⚠️ Ошибка чтения пути моделей из config.ini ({e}). Использую дефолт.", flush=True)
        models_path = "../../../models"

    # Если путь относительный, делаем его абсолютным относительно директории скрипта
    if not os.path.isabs(models_path):
        models_path = os.path.abspath(os.path.join(current_dir, models_path))

    print(f"🤖 Загрузка каскадного ансамбля моделей LightGBM из: {models_path}", flush=True)
    for agent in AGENTS_LIST:
        model_file = f"lgbm_{agent.lower()}.pkl"
        model_path = os.path.join(models_path, model_file)

        if not os.path.exists(model_path):
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Веса агента {agent} не найдены по пути: {model_path}", flush=True)
            return

        cascade_models[agent] = joblib.load(model_path)
        print(f"  -> Агент [{agent}] успешно развернут в RAM.", flush=True)

    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    symbol = "BTC/USDT"

    try:
        await asyncio.gather(
            order_book_listener(exchange, symbol),
            trades_listener(exchange, symbol),
            execution_engine(symbol, cascade_models)
        )
    finally:
        await exchange.close()
        print("🔌 Соединение с биржей закрыто.", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Робот Paper Trader остановлен пользователем.", flush=True)