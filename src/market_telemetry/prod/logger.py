import asyncio
import os
import traceback
import ccxt.pro as ccxt
import time
import configparser
from datetime import datetime, timezone, timedelta

"""
Market data logger module.
Collects order book and trade data from Bybit exchange and saves it to a CSV file.
Maintains a 32-day sliding window for the log file.
"""

# =====================================================================
# 1. ГЛОБАЛЬНЫЙ СТEЙТ С АТОМАРНЫМ КОНТРОЛЕМ RAM
# =====================================================================
class MarketState:
    """
    Global state container with atomic access control.
    Stores shared market data between producers and consumers.
    """
    def __init__(self):
        """Initializes the market state with default values and an asyncio lock."""
        self.order_book = None
        self.buy = 0.0
        self.sell = 0.0
        self.count = 0

        # Экстремумы внутри 10-секундного физического кванта
        self.current_high = -float('inf')
        self.current_low = float('inf')
        self.last_price = None

        self.lock = asyncio.Lock()
        self.seen = set()

state = MarketState()

# =====================================================================
# 2. АСИНХРОННЫЕ ПОСТАВЩИКИ ДАННЫХ (PRODUCERS)
# =====================================================================
async def order_book_producer(exchange, symbol):
    """
    Async producer that watches the order book for a given symbol.
    Updates the global state with the latest mid-price and calculates high/low within a 10s window.
    """
    while True:
        try:
            ob = await exchange.watch_order_book(symbol, limit=50)
            if not ob["bids"] or not ob["asks"]:
                continue

            # Рассчитываем Mid-Price на микро-секундном тике сокета
            mid_price = (ob["bids"][0][0] + ob["asks"][0][0]) / 2.0

            async with state.lock:
                state.order_book = ob  # Быстрое присвоение ссылки
                state.last_price = mid_price

                # Фиксируем исторический High/Low внутри 10-секундного окна
                if mid_price > state.current_high:
                    state.current_high = mid_price
                if mid_price < state.current_low:
                    state.current_low = mid_price

        except Exception:
            traceback.print_exc()
            await asyncio.sleep(2)

async def trades_producer(exchange, symbol):
    """
    Async producer that watches trades for a given symbol.
    Calculates buy/sell volumes and trade count, avoiding duplicate trade processing.
    """
    while True:
        try:
            trades = await exchange.watch_trades(symbol)

            local_buy = 0.0
            local_sell = 0.0
            local_count = 0

            for t in trades:
                tid = t.get("id") or (t.get("timestamp"), t.get("price"), t.get("amount"), t.get("side"))

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

            if local_count > 0:
                async with state.lock:
                    state.buy += local_buy
                    state.sell += local_sell
                    state.count += local_count

        except Exception:
            traceback.print_exc()
            await asyncio.sleep(2)

# =====================================================================
# 3. МАТЕМАТИКА СТАКАННЫХ ИМБАЛАНСОВ
# =====================================================================
def imbalance(ob, depth):
    """
    Calculates the order book imbalance at a specific depth.
    
    Args:
        ob (dict): Order book data.
        depth (int): Number of price levels to consider.
        
    Returns:
        float: Imbalance value (0.0 to 1.0, where 0.5 is balanced).
    """
    bids = ob["bids"][:depth]
    asks = ob["asks"][:depth]
    bv = sum(p * v for p, v in bids)  # Взвешивание по USDT-объему
    av = sum(p * v for p, v in asks)
    return round(bv / (bv + av), 4) if (bv + av) else 0.5

# =====================================================================
# 4. ПОТРЕБИТЕЛЬ-ЖУРНАЛИСТ (LOGGER CONSUMER)
# =====================================================================
async def logger_consumer(csv_file):
    """
    Async consumer that periodically (every 10s) logs market telemetry to a CSV file.
    Aggregates OHLC, imbalance, and volume data from the global state.
    """
    # Если файла нет, создаем структуру БЕЗ мертвых лейблов, но с OHLC
    if not os.path.exists(csv_file):
        # Делаем проверку директории на случай live-деплоя
        dir_name = os.path.dirname(csv_file)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(csv_file, "w", newline="\n", encoding="utf-8") as f:
            headers = [
                "timestamp", "open", "high", "low", "close",
                "imbalance_5", "imbalance_20", "imbalance_50",
                "market_delta_10s", "trade_speed_10s"
            ]
            f.write(",".join(headers) + "\n")

    last_clean_time = time.time()

    # Инициализация для плавного старта open_price
    async with state.lock:
        open_price = state.last_price if state.last_price else None

    while True:
        await asyncio.sleep(10)

        # Атомарный захват и мгновенный сброс квантовых экстремумов
        async with state.lock:
            ob = state.order_book.copy() if state.order_book else None
            buy, sell, count = state.buy, state.sell, state.count

            # Извлекаем High/Low, зафиксированные сокет-продюсером
            high_price = state.current_high
            low_price = state.current_low
            close_price = state.last_price

            # Сброс буферов объемов
            state.buy = state.sell = 0.0
            state.count = 0

            # Сброс экстремумов для следующего 10-секундного интервала
            if close_price is not None:
                state.current_high = close_price
                state.current_low = close_price
            else:
                state.current_high = -float('inf')
                state.current_low = float('inf')

        if ob is None or not ob["bids"] or not ob["asks"] or close_price is None:
            continue

        # Если на прошлом шаге open_price не определился, берем текущий close
        if open_price is None:
            open_price = close_price

        # Фильтруем редкие системные аномалии ресемплинга сокетов
        if high_price == -float('inf') or high_price < close_price:
            high_price = max(open_price, close_price)
        if low_price == float('inf') or low_price > close_price:
            low_price = min(open_price, close_price)

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        row = [
            timestamp,
            round(open_price, 2),
            round(high_price, 2),
            round(low_price, 2),
            round(close_price, 2),
            imbalance(ob, 5),
            imbalance(ob, 20),
            imbalance(ob, 50),
            round(buy - sell, 4),
            count
        ]

        with open(csv_file, 'a', encoding='utf-8') as f:
            f.write(','.join(map(str, row)) + '\n')

        # Цена закрытия текущего кванта становится ценой открытия следующего
        open_price = close_price

        # Асинхронный запуск очистки скользящего окна раз в 10 минут
        if time.time() - last_clean_time >= 10 * 60:
            async_path = os.path.abspath(csv_file)
            asyncio.create_task(asyncio.to_thread(sync_maintain_sliding_window, async_path))
            last_clean_time = time.time()

# =====================================================================
# 5. СЛУЖБА ОЧИСТКИ ОКНА ХРАНЕНИЯ (СКОЛЬЗЯЩИЕ 32 ДНЯ)
# =====================================================================
def sync_maintain_sliding_window(file_path):
    """
    Maintains a sliding window for the storage file, keeping only the last 32 days of data.
    This function is synchronous and should be run in a separate thread.
    """
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

        slice_idx = None
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            try:
                raw_timestamp_str = line.split(',')[0].strip()
                log_timestamp = datetime.fromisoformat(raw_timestamp_str)

                if log_timestamp >= cutoff_time:
                    slice_idx = i
                    break
            except Exception:
                continue

        if slice_idx is not None:
            valid_lines = lines[slice_idx:]
            if header:
                valid_lines.insert(0, header)
        else:
            valid_lines = [header] if header else []

        if len(valid_lines) < len(lines):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Очищено строк: {len(lines) - len(valid_lines)}")

    except Exception as e:
        print(f"Error inside sliding window: {e}")

# =====================================================================
# 6. СИСТЕМНАЯ ТОЧКА ВХОДА (ENTRY POINT)
# =====================================================================
async def main():
    """
    Main entry point for the logger.
    Initializes the exchange connection and starts producer and consumer tasks.
    """
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.ini")

    csv_path = "~/logger_data.csv"
    if os.path.exists(config_path):
        config.read(config_path)
        try:
            csv_path = config.get("LABELER", "csv_file_row_data", fallback=csv_path)
            print(f"📡 Загружен путь к CSV из конфига: {csv_path}")
        except Exception as e:
            print(f"⚠️ Ошибка чтения config.ini ({e}). Использую дефолт путь.")
    else:
        print(f"⚠️ Файл конфигурации не найден по пути: {config_path}. Использую дефолт путь.")

    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})

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