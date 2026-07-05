import asyncio
import csv
import os
import traceback
import ccxt.pro as ccxt
from datetime import datetime, timezone

class MarketState:
    def __init__(self):
        self.order_book = None
        self.buy = 0.0
        self.sell = 0.0
        self.count = 0
        self.lock = asyncio.Lock()
        self.seen = set()

state = MarketState()

async def order_book_producer(exchange, symbol):
    while True:
        try:
            ob = await exchange.watch_order_book(symbol, limit=50)
            async with state.lock:
                state.order_book = ob  # Быстрое присвоение ссылки вместо deepcopy
        except Exception:
            traceback.print_exc()
            await asyncio.sleep(2)

async def trades_producer(exchange, symbol):
    while True:
        try:
            trades = await exchange.watch_trades(symbol)

            # Инициализируем локальные переменные (считаем БЕЗ лока)
            local_buy = 0.0
            local_sell = 0.0
            local_count = 0

            for t in trades:
                tid = t.get("id") or (t.get("timestamp"), t.get("price"), t.get("amount"), t.get("side"))

                # Чтение/запись в state.seen внутри цикла допустимы,
                # но для идеальной точности мы не блокируем весь сборщик
                if tid in state.seen:
                    continue
                state.seen.add(tid)

                if len(state.seen) > 10000:
                    state.seen.clear()

                amt = float(t.get("amount", 0))
                if t.get("side") == "buy":
                    local_buy += amt
                elif t.get("side") == "sell":
                    local_sell += amt
                local_count += 1

            # ЗАХВАТЫВАЕМ ЛОК ОДИН РАЗ: переносим локальные агрегаты в глобальный стейт
            if local_count > 0:
                async with state.lock:
                    state.buy += local_buy
                    state.sell += local_sell
                    state.count += local_count

        except Exception:
            traceback.print_exc()
            await asyncio.sleep(2)

def imbalance(ob, depth):
    bids = ob["bids"][:depth]
    asks = ob["asks"][:depth]
    bv = sum(p * v for p, v in bids)  # Математика взвешивания по цене (USDT объем)
    av = sum(p * v for p, v in asks)
    return round(bv / (bv + av), 4) if (bv + av) else 0.5

async def logger_consumer(csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="\n") as f:
            csv.writer(f).writerow([
                "timestamp", "price", "imbalance_5", "imbalance_20",
                "imbalance_50", "market_delta_10s", "trade_speed_10s", "label_next_price"
            ])

    while True:
        await asyncio.sleep(10)

        # Атомарный захват среза данных
        async with state.lock:
            ob = state.order_book.copy() if state.order_book else None
            buy, sell, count = state.buy, state.sell, state.count
            state.buy = state.sell = 0.0
            state.count = 0

        if ob is None or not ob["bids"] or not ob["asks"]:
            continue

        price = (ob["bids"][0][0] + ob["asks"][0][0]) / 2
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        row = [
            timestamp, price,
            imbalance(ob, 5), imbalance(ob, 20), imbalance(ob, 50),
            round(buy - sell, 4), count, ""
        ]

        print(row)
        with open(csv_file, "a", newline="\n") as f:
            csv.writer(f).writerow(row)

async def main():
    exchange = ccxt.bybit({"enableRateLimit": True})
    tasks = [
        asyncio.create_task(order_book_producer(exchange, "BTC/USDT")),
        asyncio.create_task(trades_producer(exchange, "BTC/USDT")),
        asyncio.create_task(logger_consumer("../../multidim_market_data.csv"))
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await exchange.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
