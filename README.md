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
| `--host [text]`  | Specify the IP address and port for the mining pool.                                         | Yes      | `--host example.com:12345`              |
| `--use-sha`      | Enable optimizations for mining.                                                             | No       | `--use-sha`                             |
| `--threads [number]` | Specify the number of CPU threads to use for mining.                                       | No       | `--threads 4`                           |
| `--affinity`     | Use advanced CPU thread affinity settings.                                                    | No       | `--affinity`                            |
| `--gpu`          | Toggle CUDA GPU mining (if supported by hardware).                                                 | No       | `--gpu`                                  |

You can use these options when running the `3dp-miner` executable to customize its behavior according to your mining requirements.