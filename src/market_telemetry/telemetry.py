import asyncio
import ccxt.pro as ccxt  # Используем .pro версию ccxt для работы с WebSockets
import pandas as pd


async def main():
    # Инициализируем подключение к бирже Bybit (или binance)
    # Через WebSocket мы будем получать данные мгновенно, без задержек HTTP-запросов
    exchange = ccxt.bybit({"enableRateLimit": True})
    symbol = "BTC/USDT"

    print(f"📡 Датчик запущен. Подключаемся к трубе ликвидности {symbol}...")

    try:
        while True:
            # Асинхронно ждем изменения стакана (Order Book) от биржи.
            # Как только на бирже кто-то переставил лимитный ордер, биржа присылает нам дельту.
            order_book = await exchange.watch_order_book(symbol)

            # Очищаем экран терминала для красивого вывода телеметрии
            print("\033[H\033[J", end="")

            # order_book['bids'] - это покупатели (аккумуляторы давления снизу)
            # order_book['asks'] - это продавцы (заслонки сопротивления сверху)
            bids = order_book["bids"][:5]  # Берем топ-5 лучших цен покупателей
            asks = order_book["asks"][:5]  # Берем топ-5 лучших цен продавцов

            print(f"=== ТЕЛЕМЕТРИЯ СТАКАНА {symbol} ===")
            print(f"Время биржи: {order_book['datetime']}\n")

            # --- ФИЗИЧЕСКИЙ АНАЛИЗ ЗА СЛОНОК ПРОДАВЦОВ (СОПРОТИВЛЕНИЕ СВЕРХУ) ---
            print("🛑 ЗАСЛОНКИ ПРОДАВЦОВ (ASKS) - Давление сверху:")
            # Разворачиваем массив, чтобы более дорогие ордера были выше на экране
            for price, volume in reversed(asks):
                dollar_value = price * volume
                print(
                    f"  Цена: {price:<10} | Объем: {volume:<8.3f} BTC | Емкость: ${dollar_value:,.0f}"
                )

            print("-" * 50)
            print(f"   ТЕКУЩАЯ ЦЕНА (СПРЕД СВЕДЕНИЯ): {order_book['nonce']}")
            print("-" * 50)

            # --- ФИЗИЧЕСКИЙ АНАЛИЗ НАКОПИТЕЛЕЙ ПОКУПАТЕЛЕЙ (ДАВЛЕНИЕ СНИЗУ) ---
            print("🟢 НАКОПИТЕЛИ ПОКУПАТЕЛЕЙ (BIDS) - Поддержка снизу:")
            for price, volume in bids:
                dollar_value = price * volume
                print(
                    f"  Цена: {price:<10} | Объем: {volume:<8.3f} BTC | Емкость: ${dollar_value:,.0f}"
                )

            # --- МАТЕМАТИЧЕСКИЙ РАСЧЕТ БАЛАНСА СИЛ (ФИЧА) ---
            total_bid_depth = sum([p * v for p, v in order_book["bids"][:20]])
            total_ask_depth = sum([p * v for p, v in order_book["asks"][:20]])

            # Вычисляем коэффициент доминирования сил (Imbalance)
            # Если значение > 0.5 - снизу плотина толще (насосы давят вверх)
            # Если < 0.5 - сверху заслонка тяжелее (гравитация тянет вниз)
            imbalance = total_bid_depth / (total_bid_depth + total_ask_depth)

            print("\n" + "=" * 50)
            print(
                f"📊 ФИЗИЧЕСКАЯ ФИЧА (Баланс сил в стакане на 20 уровней): {imbalance:.2f}"
            )
            if imbalance > 0.55:
                print("⚠️  Сигнал датчика: Снизу плотина прочнее. Ожидается выдавливание цены вверх.")
            elif imbalance < 0.45:
                print("⚠️  Сигнал датчика: Сверху плита тяжелее. Ожидается просадка цены вниз.")
            else:
                print("⚖️  Система в равновесии. Ток распределен равномерно.")

    except Exception as e:
        print(f"Ошибка в работе датчика: {e}")
    finally:
        # Цивилизованно закрываем сокет при выходе (Ctrl+C)
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())