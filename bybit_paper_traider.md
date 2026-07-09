## Setup Service

### Open Service Config
```
vim /etc/systemd/system/bybit-paper.service
```

### ByBit Paper Service
```
[Unit]
Description=Bybit Paper Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/projects/market_telemetry/
ExecStart=/root/projects/market_telemetry/venv/bin/python3 -u bybit_paper_trader.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/bybit/bybit-paper-traider.log
StandardError=journalartSec=5

[Install]
WantedBy=multi-user.target
```

Reload and Restart
```
systemctl daemon-reload &&
systemctl restart bybit-paper.service
```

### Open Logs
```
tail -f /var/log/bybit/bybit-paper-traider.log
```

### Download Log
```
scp -i ~/.ssh/key2 root@157.180.16.28:/var/log/bybit/bybit-paper-traider.log ~/tmp/bybit-paper-traider.log 
```


## Setup Log File Rotation

