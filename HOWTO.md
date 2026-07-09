# How To

## Оглавление
* [Download all row csv data from Hetzner](#download-all-row-csv-data-from-hetzner)
* [Create big data fileset](#create-big-data-fileset)
* [Train Model and Run Backtester](#train-model-and-run-backtester)
* [Upload all Necessary Files on Hetzner](#upload-all-necessary-files-on-hetzner)
* [Manage Paper Tariding Service](#manage-paper-tariding-service)
* [Run any Python File](#how-to-run-any-python-file)
___

## Local Work
___

### Download all row csv data from Hetzner

Last file
```shell
scp -i ~/.ssh/key2 root@157.180.16.28:/root/multidim_market_data.csv ~/projects/python/market_telemetry/data/hetzner
```
Archives
```shell
scp -i ~/.ssh/key2 root@157.180.16.28:/root/multidim_market_data.csv.*.gz ~/projects/python/market_telemetry/data/hetzner
```

### Create big data fileset
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry &&
source ~/projects/python/market_telemetry/.venv/bin/activate &&
python prepare_big_data.py
```

### Train Model and Run Backtester
Go to the project folder
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry
```

activate envinronment
```shell
source ~/projects/python/market_telemetry/.venv/bin/activate
```

And start scripts
```shell
python labeler.py && python lgbm_train.py && python tester.py
```

Then visualize it
```shell
python visualize_signals.py
```

### Download the csv file
```shell
scp -i ~/.ssh/key2 root@157.180.16.28:/root/multidim_market_data.csv ~/projects/python/market_telemetry/data/hetzner
```

### Upload all Necessary Files on Hetzner


Upload balance checker
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 balance_check.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload live trader
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 bybit_live_trader.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload paper trader
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod && 
scp -i ~/.ssh/key2 bybit_paper_trader.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload config IMPORTANT NOTE ALL CREDS WILL BE OVERRIDDEN BY MASK
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 config.ini root@157.180.16.28:/root/projects/market_telemetry
```

Upload regme
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 market_regime.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload model
```shell
cd ~/projects/python/market_telemetry/data &&
scp -i ~/.ssh/key2 lgbm_live_model.pkl root@157.180.16.28:/root/projects/market_telemetry
```

Upload Logger
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 multidim_logger.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload labeler
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 labeler.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload Trainer
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 lgbm_train.py root@157.180.16.28:/root/projects/market_telemetry
```

Upload pipeline shell
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 run_nightly_pipeline.sh root@157.180.16.28:/root/projects/market_telemetry
```

### Manage Paper Tariding Service


Logs Paper Trader
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


### Run any Python File in Prod

```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
source ~/projects/python/market_telemetry/.venv/bin/activate &&
python any_file.py
````



## Remote Work

### Download all log files of papaer traider

Enter Hetzner
```
ssh -i ~/.ssh/key2 root@157.180.16.28
``` 

Create log files on hetzner
```
sudo journalctl -u bybit-paper.service > alllogs.txt
```

Download the log file
```bash
scp -i ~/.ssh/key2 root@157.180.16.28:/root/alllogs.txt ~/logs/
```


### Force Train Model

Ssh to Hetzner
```
ssh -i ~/.ssh/key2 root@157.180.16.28
```

Run rebuild pipeline
```
/root/projects/python/run_nightly_pipeline.sh
```