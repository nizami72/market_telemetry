# Bybit Market Data Logger

This service is designed to collect real-time market data from the Bybit exchange, specifically for the BTC/USDT pair. It captures order book depth and trade executions, processing them into a multidimensional format for further analysis and machine learning labeling.

## Overview

The logger performs the following tasks:
- **Order Book Monitoring**: Watches the top 50 levels of the order book to calculate mid-price and imbalances.
- **Trade Monitoring**: Tracks individual trades to aggregate buy/sell volumes and trade counts.
- **Data Persistence**: Saves aggregated data every 10 seconds to a CSV file.
- **Maintenance**: Implements a 32-day sliding window to keep the log file size manageable.

## Deployment to Remote Server

To deploy the logger to the remote Hetzner machine, follow these steps.

### SSH Access
The remote server is available at:
- **IP**: `157.180.16.28`
- **User**: `root`
- **Identity File**: `/home/nizami/.ssh/key2`

### Uploading Files
Use `scp` to upload the necessary source files to the remote server:

```bash
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i /home/nizami/.ssh/key2 -r logger.py root@157.180.16.28:/root/projects/market_telemetry
```

## Systemd Service Configuration

To ensure the logger runs continuously and restarts automatically on failure, it should be managed as a systemd service.

### Service File Content
Create the service file at on the remote machine:
```
vim /etc/systemd/system/bybit-logger.service
```

and insert the following content:

```ini
[Unit]
Description=Bybit Market Data Logger Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/projects/market_telemetry
ExecStart=/root/projects/market_telemetry/venv/bin/python logger.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/bybit/logger.log
StandardError=append:/var/log/bybit/logger.log

[Install]
WantedBy=multi-user.target
```


*Note: Ensure the log directory exists on the remote machine:*
```bash
ls -p /var/log/bybit
```

Rload the systemd daemon:
```bash
sudo systemctl daemon-reload
```

### Managing the Service

Use the following commands to manage the logger service:

- **Start the service**:
  ```bash
  systemctl start bybit-logger
  ```

- **Stop the service**:
  ```bash
  systemctl stop bybit-logger
  ```

- **Restart the service**:
  ```bash
  systemctl restart bybit-logger
  ```

- **Enable auto-start on boot**:
  ```bash
  systemctl enable bybit-logger
  ```

- **Check service status**:
  ```bash
  systemctl status bybit-logger
  ```

- **View real-time logs**:
  ```bash
  tail -f /var/log/bybit/logger.log
  ```
- **View real-time logged data**:
```bash
tail -f /root/projects/market_telemetry/data/raw_data.csv
```
