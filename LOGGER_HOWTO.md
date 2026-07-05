


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