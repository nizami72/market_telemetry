# How To

## Оглавление
* [How to download all row csv data from Hetzner](#how-to-download-all-row-csv-data-from-hetzner)
* [How to create big data fileset](#how-to-create-big-data-fileset)
* [How to Train Model and Run Backtester](#how-to-train-model-and-run-backtester)
* [How to Upload all Necessary Files on Hetzner](#how-to-upload-all-necessary-files-on-hetzner)
* [How to Manage Paper Tariding Service](#how-to-manage-paper-tariding-service)
___

## Local Work
___

### How to download all row csv data from Hetzner

Last file
```shell
scp -i /home/nizami/.ssh/key2 root@157.180.16.28:/root/multidim_market_data.csv /home/nizami/projects/python/market_telemetry/data/hetzner
```
Archives
```shell
scp -i /home/nizami/.ssh/key2 root@157.180.16.28:/root/multidim_market_data.csv.*.gz /home/nizami/projects/python/market_telemetry/data/hetzner
```

### How to create big data fileset
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry &&
source /home/nizami/projects/python/market_telemetry/.venv/bin/activate &&
python prepare_big_data.py
```

### How to Train Model and Run Backtester
Go to the project folder
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry
```

activate envinronment
```shell
source /home/nizami/projects/python/market_telemetry/.venv/bin/activate
```

And start scripts
```shell
python multidim_labeler.py && python lgbm_train.py && python lgbm_backtester.py
```

Then visualize it
```shell
python visualize_signals.py
```

### How to download the csv file
```shell
scp -i /home/nizami/.ssh/key2 root@157.180.16.28:/root/multidim_market_data.csv /home/nizami/projects/python/market_telemetry/data/hetzner
```

### How to Upload all Necessary Files on Hetzner


Upload balance checker
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i /home/nizami/.ssh/key2 balance_check.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload live trader
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i /home/nizami/.ssh/key2 bybit_live_trader.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload paper trader
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod && 
scp -i /home/nizami/.ssh/key2 bybit_paper_trader.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload config IMPORTANT NOTE ALL CREDS WILL BE OVERRIDDEN BY MASK
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i /home/nizami/.ssh/key2 config.ini root@157.180.16.28:/root/projects/market_telemetry
```


Upload regme
```shell
cd /home/nizami/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i /home/nizami/.ssh/key2 market_regime.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload model
```shell
cd /home/nizami/projects/python/market_telemetry/data &&
scp -i /home/nizami/.ssh/key2 lgbm_live_model.pkl root@157.180.16.28:/root/projects/market_telemetry
```
___

### How to Manage Paper Tariding Service


Logs
```shell
  sudo journalctl -u bybit-paper.service -f -n 50
```

Enable Service
```shell
  sudo systemctl enable bybit-paper.service
```

Restart Service
```shell
  sudo systemctl restart bybit-paper.service && sudo systemctl status bybit-paper.service
```

Start Service
```shell
  sudo systemctl start bybit-paper.service
```

Status Service
```shell
  sudo systemctl status bybit-paper.service
```

Stop Service
```shell
  sudo systemctl stop bybit-paper.service
```
Open Service file
```shell
  sudo vim /etc/systemd/system/bybit-paper.service
```

## Remote Work