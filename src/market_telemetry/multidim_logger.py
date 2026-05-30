import asyncio
import os
import ccxt.pro as ccxt
import pandas as pd

# --- ГЛОБАЛЬНЫЕ ХРАНИЛИЩА ДЛЯ ОБМЕНА ДАННЫМИ МЕЖДУ ПОТОКАМИ ---
latest_order_book = None
trade_volume_buy_10s = 0.0
trade_volume_sell_10s = 0.0
trade_count_10s = 0


# 1. ПРОДУЦЕНТ СТАКАНА: непрерывно слушает намерения игроков
async def order_book_producer(exchange, symbol):
    global latest_order_book
    while True:
        try:
            # Запрашиваем лимит 50, разрешенный биржей Bybit
            latest_order_book = await exchange.watch_order_book(symbol, limit=50)
        except Exception as e:
            print(f"❌ Ошибка WebSocket стакана: {e}")
            await asyncio.sleep(2)


# 2. ПРОДУЦЕНТ ЛЕНТЫ СДЕЛОК: непрерывно замеряет кинетический ток
async def trades_producer(exchange, symbol):
    global trade_volume_buy_10s, trade_volume_sell_10s, trade_count_10s
    while True:
        try:
            trades = await exchange.watch_trades(symbol)
            for trade in trades:
                trade_count_10s += 1
                volume = trade["amount"]  # Объем сделки в BTC
                if trade["side"] == "buy":
                    trade_volume_buy_10s += volume  # Рыночный ордер на покупку
                else:
                    trade_volume_sell_10s += volume  # Рыночный ордер на продажу
        except Exception as e:
            print(f"❌ Ошибка WebSocket ленты сделок: {e}")
            await asyncio.sleep(2)


# 3. ПОТРЕБИТЕЛЬ (ЗАПИСЬ): делает снимок датчиков каждые 10 секунд
async def logger_consumer(symbol, csv_file):
    global latest_order_book, trade_volume_buy_10s, trade_volume_sell_10s, trade_count_10s

    print(
        f"📡 Многомерный логгер запущен. Сбор данных в {csv_file} начал ход..."
    )

    # Инициализация структуры CSV, если файла нет
    if not os.path.exists(csv_file):
        headers = [
            "timestamp",
            "price",
            "imbalance_5",
            "imbalance_20",
            "imbalance_50",
            "market_delta_10s",
            "trade_speed_10s",
            "label_next_price",
        ]
        pd.DataFrame(columns=headers).to_csv(csv_file, index=False)

    while True:
        # Ждем ровно 10 секунд для формирования физического кванта времени
        await asyncio.sleep(10)

        # Проверяем, успели ли продуценты собрать первые данные
        if latest_order_book is None or trade_count_10s == 0:
            print("⏳ Ожидание первого наполнения буфера данных от биржи...")
            continue

        timestamp = pd.Timestamp.now()
        current_price = (
                                latest_order_book["bids"][0][0] + latest_order_book["asks"][0][0]
                        ) / 2

        # --- РАСЧЕТ ПРОСТРАНСТВЕННЫХ ФИЧ (СТАКАН) ---
        def calc_imbalance(depth):
            b_depth = sum([p * v for p, v in latest_order_book["bids"][:depth]])
            a_depth = sum([p * v for p, v in latest_order_book["asks"][:depth]])
            return round(b_depth / (b_depth + a_depth), 4) if (b_depth + a_depth) > 0 else 0.5

        imb_5 = calc_imbalance(5)
        imb_20 = calc_imbalance(20)
        imb_50 = calc_imbalance(50)

        # --- РАСЧЕТ КИНЕТИЧЕСКИХ ФИЧ (ЛЕНТА) ---
        # Дельта = Покупки минус Продажи. Плюс — давят покупатели, Минус — продавцы
        market_delta = trade_volume_buy_10s - trade_volume_sell_10s
        speed = trade_count_10s

        # --- СБОР СТРОКИ ДАТА СЕТА ---
        new_row = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "imbalance_5": imb_5,
            "imbalance_20": imb_20,
            "imbalance_50": imb_50,
            "market_delta_10s": round(market_delta, 4),
            "trade_speed_10s": speed,
            "label_next_price": None,
        }

        # Пишем в лог терминала текущий срез физики
        print(
            f"[{timestamp.strftime('%H:%M:%S')}] Цена: {current_price:<8} | "
            f"Imb(5/20/50): {imb_5:.2f}/{imb_20:.2f}/{imb_50:.2f} | "
            f"Delta: {market_delta:>7.3f} BTC | Speed: {speed} т/10с"
        )

        # СБРОС И КЛИНИНГ БУФЕРА ЛЕНТЫ ДЛЯ СЛЕДУЮЩИХ 10 СЕКУНД
        trade_volume_buy_10s = 0.0
        trade_volume_sell_10s = 0.0
        trade_count_10s = 0

        # Сохранение в CSV
        pd.DataFrame([new_row]).to_csv(csv_file, mode="a", header=False, index=False)


async def main():
    # Создаем асинхронный инстанс Bybit
    exchange = ccxt.bybit({"enableRateLimit": True})
    symbol = "BTC/USDT"
    csv_file = "../../multidim_market_data.csv"

    # Запускаем параллельные задачи (Workers) в фоне общего Event Loop
    task_book = asyncio.create_task(order_book_producer(exchange, symbol))
    task_trades = asyncio.create_task(trades_producer(exchange, symbol))
    task_logger = asyncio.create_task(logger_consumer(symbol, csv_file))

    # Держим main() активным, пока работает логгер
    await task_logger


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Сбор данных остановлен пользователем.")