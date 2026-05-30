import asyncio
import os
import ccxt.pro as ccxt
import pandas as pd


async def main():
    # Инициализируем асинхронный Bybit
    exchange = ccxt.bybit({"enableRateLimit": True})
    symbol = "BTC/USDT"
    csv_file = "../../market_data.csv"

    print(f"📡 Logger запущен. Начинаю сбор базы данных в файл {csv_file}...")

    # Если файла еще нет, создаем структуру базы данных
    if not os.path.exists(csv_file):
        df_init = pd.DataFrame(
            columns=["timestamp", "price", "imbalance_20", "label_next_price"]
        )
        df_init.to_csv(csv_file, index=False)

    try:
        while True:
            # ИСПРАВЛЕНО: watch_order_book вместо watch_book
            order_book = await exchange.watch_order_book(symbol, limit=50)

            # Извлекаем физические параметры баланса сил
            current_price = (
                                    order_book["bids"][0][0] + order_book["asks"][0][0]
                            ) / 2
            timestamp = pd.Timestamp.now()

            # Считаем емкость плотин (на 20 уровней вглубь стакана)
            total_bid_depth = sum([p * v for p, v in order_book["bids"][:20]])
            total_ask_depth = sum([p * v for p, v in order_book["asks"][:20]])

            if (total_bid_depth + total_ask_depth) == 0:
                continue

            imbalance = total_bid_depth / (total_bid_depth + total_ask_depth)

            # Формируем строку для записи телеметрии
            new_row = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "price": current_price,
                "imbalance_20": round(imbalance, 4),
                "label_next_price": None,  # Сюда "машина времени" позже проставит исход
            }

            # Вывод текущего состояния в консоль
            print(
                f"[{timestamp.strftime('%H:%M:%S')}] Цена: {current_price:<9} | Imbalance: {imbalance:.3f} -> Запись в CSV"
            )

            # Дописываем строку данных в конец файла без перезаписи всего CSV
            df_row = pd.DataFrame([new_row])
            df_row.to_csv(csv_file, mode="a", header=False, index=False)

            # Засыпаем на 10 секунд, освобождая поток для Event Loop
            await asyncio.sleep(10)

    except Exception as e:
        print(f"⚠️ Ошибка логгера: {e}")
    finally:
        # Обязательно закрываем сетевое WebSocket-соединение
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
