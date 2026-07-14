## LABELER


### Launch Locally
```bash
python3 ~/projects/python/market_telemetry/src/market_telemetry/prod/labeler.py ~/projects/python/market_telemetry/data/csv/raw_data_2026-06.csv
```


### Upload labeler
```shell
cd ~/projects/python/market_telemetry/src/market_telemetry/prod &&
scp -i ~/.ssh/key2 labeler.py root@157.180.16.28:/root/projects/market_telemetry labeler.py
```