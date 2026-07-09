


### Check status
```
sudo systemctl status telemetry-logger.service
```

### Reload Service
```
sudo systemctl reload-or-restart telemetry-logger.service
```

### Logs journalctl
```
sudo journalctl -u telemetry-logger.service
```

### Logs Tail
```
tail -f 100 /root/multidim_market_data.csv
```

### Logs with Vim
```
vim /root/multidim_market_data.csv
```


### Upload multidim_logger.py File to Hetzner
```bash
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i /home/nizami/.ssh/key2 multidim_logger.py root@157.180.16.28:/root/projects/market_telemetry
```

### Как убрать ^M из уже записанного файла прямо сейчас?
Чтобы очистить файл от уже накопившихся символов ^M на сервере, выполни одну команду в консоли Hetzner:

```Bash
sed -i 's/\r//g' multidim_market_data_new.csv
```
Эта команда мгновенно удалит все возвраты каретки из файла, сделав его формат чисто линуксовым.

Как скачть лог логгера
```
ssh -i ~/.ssh/key2 root@157.180.16.28
```

```
sudo journalctl -u telemetry-logger.service > logger.log
```

```
scp -i ~/.ssh/key2 root@157.180.16.28:/root/logger.log ~/Desktop/
```

### Logger Service File

/etc/systemd/system/telemetry-logger.service

```
[Unit]
Description=Market Telemetry Big Data Logger
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/projects/market_telemetry
ExecStart=/root/projects/market_telemetry/venv/bin/python multidim_logger.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/telemetry-logger.log
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### SETUP Logger Logrotate File, Not implemented yet
```
sudo vim /etc/logrotate.d/telemetry-logger
```

```
/var/log/telemetry-logger.log {
daily               # Ротация каждый день
rotate 7            # Хранить логи только за последние 7 дней (остальное удалять)
compress            # Сжимать старые логи в .gz (экономит 90% места)
delaycompress       # Не сжимать самый свежий вчерашний лог
missingok           # Не выдавать ошибку, если файла нет
notifempty          # Не делать ротацию, если файл пустой
copytruncate        # Отрезать лог «на лету», не останавливая Python-скрипт
}
```