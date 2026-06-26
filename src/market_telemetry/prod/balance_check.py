import json
import configparser
from pybit.unified_trading import HTTP

config = configparser.ConfigParser()
config.read("config.ini")

API_KEY = config.get("API_KEYS", "bybit_api_key")
API_SECRET = config.get("API_KEYS", "bybit_api_secret")

try:
    # Инициализируем сессию Bybit строго в режиме Sandbox (testnet=True)
    session = HTTP(
        testnet=True,
        api_key=API_KEY,
        api_secret=API_SECRET,
    )

    # Запрашиваем баланс Единого Торгового Аккаунта (UNIFIED)
    response = session.get_wallet_balance(accountType="UNIFIED")
    
    print("\n" + "="*50)
    print("🟢 ДОМАШНИЙ SMOKE TEST ПРОЙДЕН УСПЕШНО!")
    print("="*50)
    print(f"STATUS: 200 OK")
    print("RESPONSE BALANCE:")
    print(json.dumps(response, indent=2))
    print("="*50 + "\n")

except Exception as e:
    print("\n" + "="*50)
    print("❌ ОШИБКА АВТОРИЗАЦИИ В TESTNET:")
    print(e)
    print("="*50 + "\n")

