# 3dp-miner
Full package of CPU + GPU miner

* The main changes for optimizations are in the find_top_std_3 function (cpu) and find_top_std_2 (gpu)
* Both implementations are for grid2d v3
* The miner is only for solo mining, using [easy3dp node](https://github.com/easy-3dp/3DP)

## Usage on Ubuntu 22.04 (WSL Supported)
To run the `3dp-miner`, open a terminal or command prompt and navigate to the directory where the executable is located. Then, use the following syntax:

```bash
./3dp-miner --host example.com:12345 --use-sha
```

| Option         | Description                                                                                   | Required | Example Usage                           |
|----------------|-----------------------------------------------------------------------------------------------|----------|-----------------------------------------|
| `--host [text]`  | Specify the IP address and port for the [easy3dp node](https://github.com/easy-3dp/3DP).                                         | Yes      | `--host example.com:12345`              |
| `--use-sha`      | Enable optimizations for mining.                                                             | No       | `--use-sha`                             |
| `--threads [number]` | Specify the number of CPU threads to use for mining.                                       | No       | `--threads 4`                           |
| `--affinity`     | Use advanced CPU thread affinity settings.                                                    | No       | `--affinity`                            |
| `--gpu`          | Toggle CUDA GPU mining (if supported by hardware).                                                 | No       | `--gpu`                                  |

You can use these options when running the `3dp-miner` executable to customize its behavior according to your mining requirements.

## Hashrate Comparison Chart

| Hardware    | Approx. Hashrate (KH/s) | Approx. Old Hashrate (KH/s) |
|-------------|-----------------|-----------------------|
| AMD EPYC 7742     | 180.5 | - |
| Ryzen 7900x       | 165   | - |
| Ryzen 5950x       | 159   | 11    |
| Ryzen 5900x       | 122   | 8.5   |
| RTX 4090          | 69    | N/A   |
| RTX 4070          | 35    | N/A   |
| Ryzen 3600x       | 58    | 5     |
| i7-6800k          | 34    | 2     |
|-------------|-----------------|-----------------------|
| AMD EPYC  7v12    | 260   | 20    |
| AMD EPYC  7642    | 160   | -     |
| AMD 5800H         | 85    | -     |
| AMD 5900X         | 120   | -     |
| AMD 5800X         |  80   | -     |
| AMD 3900X         | 90   | -     |
| RTX 3070          | 14    | -     |
| RTX 3060ti        | 13    | -     |
| nvidia 90hx       | 15    | -     |
