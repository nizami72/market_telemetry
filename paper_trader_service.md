# Paper Traider Sevice

## Оглавление
* [Upload paper trader on Hetzner](#upload-paper-trader-on-hetzner)
* 🚀 [Manual Start](manual-start)
* 
___

### Шаг 1. Upload paper trader on Hetzner

#### Go to the Sources Folder
```
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod
```

#### Upload the Paper Trader File
```
scp -i /home/nizami/.ssh/key2 bybit_paper_trader.py root@157.180.16.28:/root/projects/market_telemetry
```

#### Restart the Paper Service on Hetzner
```
sudo systemctl restart bybit-paper.service &&
sudo systemctl status bybit-paper.service
```

#### Check the Service Status
```
sudo journalctl -u bybit-paper.service -f -n 50
```

### 🚀 Manual Start


(venv) root@ubuntu-4gb-hel1-3:~/projects/market_telemetry# cd projects/market_telemetry/
root@ubuntu-4gb-hel1-3:~/projects/market_telemetry# source venv/bin/activate

Запустим скрипт напрямую через Python, чтобы убедиться, что он успешно импортирует библиотеки, находит модель и подключается к WebSocket:

```python3 paper_trader.py```

🎯 Что ты должен увидеть в консоли:
Сообщение о чтении конфига: 📡 Загрузка конфигурации из: /root/projects/market_telemetry/config.ini.

Лог инициализации: [INIT] Стартовый виртуальный баланс: $10000.00 USDT.

```
Сообщение от ИИ: 🤖 Загружаю модель ИИ из: /root/projects/market_telemetry/lgbm_market_model.pkl....

Подключение: [CONNECT] Подключение к реальному стриму Bybit WebSocket (Linear)....
```

Первые 5 минут (30 тиков по 10 секунд) робот будет писать:
`⏳ Накапливаю RAM-буфер истории для расчета макро-фич... (1/30).`

Инфо: Как только счетчик дойдет до 30/30, RAM-буфер заполнится, и робот начнет каждые 10 секунд выводить в консоль текущие предсказания модели и вероятности классов Up/Flat/Down.

### 📦 Шаг 3. Перевод в режим фоновой службы (Служба 24/7)
Чтобы робот не закрывался при отключении от SSH и работал непрерывно, давай оформим его как системный демон Linux.

Открой файл новой системной службы

```
sudo vim /etc/systemd/system/bybit-paper.service
```

Вставь туда следующий стандартный прод-конфиг:

```
[Unit]
Description=Bybit Paper Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/projects/market_telemetry/
ExecStart=/root/projects/market_telemetry/venv/bin/python3 -u paper_trader.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Сохрани файл (Esc, :wq, Enter).

Перезагрузи менеджер системных служб, активируй автозапуск бота при старте сервера и запусти его прямо сейчас:

```
sudo systemctl daemon-reload
```

```
sudo systemctl enable bybit-paper.service
```

```
sudo systemctl start bybit-paper.service
```
### Шаг 4. Управление и Мониторинг
Теперь твой Paper Trader автономен. Вот три главные команды для контроля за его состоянием:
Проверить, работает ли бот прямо сейчас:

```bash
sudo systemctl status bybit-paper.service
```
Посмотреть живой поток предсказаний ИИ (вывод в консоль на лету):

```
sudo journalctl -u bybit-paper.service -f -n 50
```

Проверить изолированный журнал только совершенных сделок и баланса:

```
tail -f /root/projects/market_telemetry/paper_trading.log
```

### Logs

```
tail -fn 100 /var/log/bybit/bybit-paper-traider.log
```

#### Download Logs from Bybit to Local Machine
```shell
scp -i /home/nizami/.ssh/key2 root@157.180.16.28:/var/log/bybit/bybit-paper-traider.log ~/tmp/logs
```