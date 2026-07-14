## 🛠️ Документация: Динамический анализатор `market_regime.py`

### 1. Назначение и функции файла

`market_regime.py` — это изолированный фоновый бэкэнд-модуль (Watcher), отвечающий за **динамическое определение текущей фазы рынка** («Штиль» или «Шторм»).

**Что он делает:**

* Считывает исторический лог сырых тиков Биткоина (путь к CSV берется напрямую из `config.ini`).


* Агрегирует тиковые данные за последние **24 часа** в 15-минутные свечи OHLC.


* Рассчитывает средний размах (волатильность) одной свечи ($ATR_{24h}$).


* **Замыкает контур (MLOps):** На основе авто-математики ATR *на лету переписывает центральный файл `config.ini` прямо на диске Hetzner VPS*, жестко хардкодя актуальные метаданные как для разметчика (`noise_threshold`, `data_thinning_step`), так и для торгового робота (`confidence_threshold`, `tp_sl_size`).



### 2. Запуск и интервал автоматизации

* **Как запускается:** Скрипт выполняется линейно в фоновом режиме операционной системы. Он производит мгновенный расчет в RAM, фиксирует данные на диск и полностью выгружается из памяти (нулевая утечка ресурсов VPS).


* **Где находится интервал:** Интервал запуска полностью контролируется планировщиком задач Linux (`systemd.timer` или `cron`) на Hetzner VPS. Оптимальный промышленный интервал для среднесрочной стратегии — **раз в 1 час** или **раз в 15 минут**.

### 3. Upload to Hetzner

```bash
cd ~/projects/python/market_telemetry/src/market_telemetry/prod && 
scp -i /home/nizami/.ssh/key2 market_regime.py root@157.180.16.28:/root/projects/market_telemetry
```


---

## ⚙️ Развертывание: Systemd-сервис и Таймер (Режим Cron)

Поскольку твоя система построена по канонам монолитных демонов Linux, вместо классического `crontab` мы оформим расписание через связку `systemd.service` + `systemd.timer`. Это гарантирует отказоустойчивость, автоматический подъем при сбое и чистый вывод логов в `journald`.

### 1. Создание конфигурации службы (`market-regime.service`)

Открой терминал VPS и создай файл службы через Vim:

```bash
sudo vim /etc/systemd/system/market-regime.service

```

Вставь следующий production-конфиг (скорректируй пути под своего пользователя):

```ini
[Unit]
Description=Market Regime Volatility Watcher
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/root/projects/market_telemetry/src/market_telemetry
ExecStart=/root/projects/market_telemetry/venv/bin/python market_regime.py
User=root

[Install]
WantedBy=multi-user.target

```

### 2. Создание таймера расписания (`market-regime.timer`)

Теперь создаем файл таймера, который будет дергать наш сервис **каждый час** (интервал задается в строке `OnCalendar`):

```bash
sudo vim /etc/systemd/system/market-regime.timer

```

Вставь конфигурацию расписания:

```ini
[Unit]
Description=Run Market Regime Watcher Every Hour

[Timer]
# Запуск в начале каждого часа (например, 13:00, 14:00 и т.д.)
OnCalendar=*-*-* *:00:00
# Предохранитель: если сервер лежал, запустить пропущенную задачу сразу после старта
Persistent=true

[Install]
WantedBy=timers.target

```

---

## ⚡ Системные команды управления (CMD)

После сохранения файлов в Vim, примени команды cmd для инициализации и запуска контура автоматизации:


### Активация и запуск (Выполнить один раз)

#### Перезагрузить менеджер конфигураций systemd, чтобы он увидел новые файлы
```bash
sudo systemctl daemon-reload
```

#### Включить автозапуск таймера при загрузке сервера и запустить его прямо сейчас
```bash
sudo systemctl enable market-regime.timer --now
```

### Команды администрирования

* **Проверить статус расписания и время следующего запуска:**
```bash
sudo systemctl status market-regime.timer

```


* **Принудительный ручной запуск расчета (вне расписания):**
```bash
sudo systemctl start market-regime.service

```


* **Остановить автоматическое расписание:**
```bash
sudo systemctl stop market-regime.timer

```


* **Временно перезапустить таймер:**
```bash
sudo systemctl restart market-regime.timer

```


* **Посмотреть живой системный журнал и логи принтов (journalctl):**
```bash
journalctl -u market-regime.service -n 50 --no-pager

```



Контур полностью готов. `market_regime.py` будет автономно хардкодить `config.ini` на диске Hetzner, а твой асинхронный `paper_trader.py` подхватит изменения налету каждые 10 секунд!
