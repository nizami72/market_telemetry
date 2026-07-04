import os
import requests
import asyncio
import aiohttp
import configparser

def _load_tg_config():
    """Внутренний помощник для безопасного чтения конфига."""
    config_file = "config.ini"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"❌ Модуль TG: Конфигурационный файл {config_file} не найден!")

    cfg = configparser.ConfigParser()
    cfg.read(config_file)

    if not cfg.has_section("TELEGRAM"):
        raise KeyError("❌ Модуль TG: В config.ini отсутствует секция [TELEGRAM]")

    token = cfg.get("TELEGRAM", "bot_token")
    chat_id = cfg.get("TELEGRAM", "chat_id")
    return token, chat_id

async def send_telegram_alert_async(text: str):
    """Асинхронная отправка для live-роботов и WebSocket-шлюзов (без блокировки потока)."""
    try:
        token, chat_id = _load_tg_config()
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=5) as response:
                if response.status != 200:
                    print(f"⚠️ Ошибка асинхронного TG: {response.status}")
    except Exception as e:
        print(f"❌ Сбой асинхронной отправки TG-алерта: {e}")

def send_telegram_alert_sync(text: str):
    """Синхронная отправка для линейных скриптов, Cron-задач и анализатора режимов."""
    try:
        token, chat_id = _load_tg_config()
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

        response = requests.post(url, json=payload, timeout=5)
        if response.status_code != 200:
            print(f"⚠️ Ошибка синхронного TG: {response.status_code}")
    except Exception as e:
        print(f"❌ Сбой синхронной отправки TG-алерта: {e}")