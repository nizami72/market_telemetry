import asyncio
import os
import traceback
import ccxt.pro as ccxt
import time
from datetime import datetime, timezone, timedelta

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
            headers = [
                "timestamp", "price", "imbalance_5", "imbalance_20",
                "imbalance_50", "market_delta_10s", "trade_speed_10s", "label_next_price"
            ]
            f.write(",".join(headers) + "\n")

    last_clean_time = time.time()

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

        price = round((ob["bids"][0][0] + ob["asks"][0][0]) / 2, 2)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        row = [
            timestamp, price,
            imbalance(ob, 5), imbalance(ob, 20), imbalance(ob, 50),
            round(buy - sell, 4), count, ""
        ]

        print(row)
        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write(','.join(map(str, row)) + '\n')

        # 2. Асинхронный запуск очистки раз в 10 минут (не блокирует сбор данных)
        if time.time() - last_clean_time >= 10 * 60:
            asyncio.create_task(asyncio.to_thread(sync_maintain_sliding_window, csv_file))
            last_clean_time = time.time()


# Вынесли тяжелую очистку файла в отдельную синхронную функцию
def sync_maintain_sliding_window(file_path):
    WINDOW_SECONDS = timedelta(days=32, seconds=10)
    current_time = datetime.now(timezone.utc)
    cutoff_time = current_time - WINDOW_SECONDS

    if not os.path.exists(file_path):
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return

        has_header = "timestamp" in lines[0]
        header = lines[0] if has_header else None
        start_idx = 1 if has_header else 0

        # Ищем индекс первой строки, которая ЗАХОДИТ в скользящее окно
        slice_idx = None
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            try:
                raw_timestamp_str = line.split(',')[0].strip()
                # Если парсинг упадет, управление уйдет в except
                log_timestamp = datetime.fromisoformat(raw_timestamp_str)

                # Как только нашли строку из будущего/актуального окна:
                if log_timestamp >= cutoff_time:
                    slice_idx = i
                    break
            except Exception:
                # Если строка битая, пропускаем её в поиске точки отсечения
                continue

        # Формируем новый массив строк
        if slice_idx is not None:
            # Берем всё начиная с slice_idx до самого конца файла
            valid_lines = lines[slice_idx:]
            if header:
                valid_lines.insert(0, header)
        else:
            # Если ни одна строка не подошла под условия окна, оставляем только заголовок
            valid_lines = [header] if header else []

        # Перезаписываем файл ТОЛЬКО если реально что-то удалили
        if len(valid_lines) < len(lines):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Очищено строк: {len(lines) - len(valid_lines)}")

    except Exception as e:
        print(f"Error: {e}")


async def main():
    exchange = ccxt.bybit({"enableRateLimit": True})
    csv_path = "../../multidim_market_data.csv"
    # csv_path = "multidim_market_data.csv"
    tasks = [
        asyncio.create_task(order_book_producer(exchange, "BTC/USDT")),
        asyncio.create_task(trades_producer(exchange, "BTC/USDT")),
        asyncio.create_task(logger_consumer(csv_path))
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
