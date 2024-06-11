---
title: "Active Data Acquisition in Autonomous Driving Simulation"
collection: teaching
type: "Personal project"
permalink: /teaching/2015-spring-teaching-5
venue: "CARLA"
date: 2023-08-12
location: "Python, ROS, CARLA"
---

Data collection tools with active data acquisition algorithm implemented for CARLA Simulator.

![](../images/carla/s4.png)

You can find the source code of Cat Chat on my [GitHub](https://github.com/Kazawaryu/CARLA_ADA) repository.

## News

**2024.1.10** Fix label tool and format helper, adapter for pcdet.

**2023.12.20** Finish L2 and L4.

**2023.9.16** Finish L1 and L3 part, at ``./label_tools/lidar_tool``.

**2023.6.23** Finish the main framework design, and basic framework of algorithm.

## Dependency

- Ubuntu 20.04

- Python 3.8+

- CARLA 0.9.15

- CUDA 11.7

## Installation

1. Download Carla version 0.9.15 [here](https://github.com/carla-simulator/carla).

2. Build conda environment
   
   ```shell
   cd CARLA_ADA
   conda create -n carla38 python=3.8 & conda activate carla38
   
   pip3 install -r requirements.txt
   ```

## Usage

### Data Recording

1. Start carla, at carla root directory
   
   ```shell
   ./CarlaUE4.sh
   ```

2. Start recorder, execute the command in the root directory
   
   ```shell
   python3 data_recorder.py
   ```
   
   The configuration can be changed by modifying the files in the folder `config`.

3. Make data (image, lidar) to a dataset with labels
   
   ```shell
   python format_helper.py -s <raw_data_directory>
   ```

## Data and Result

![](../images/carla/performance.png)

## Contributing

Thank you for your interest in contributing to this project! Contributions are highly appreciated and help improve the project for everyone. If you have any questions or need further assistance, please feel free to open an issue.

## Acknowledgments

- [CARLA Simulator](https://carla.org/)
- [CARLA Ros Bridge](https://github.com/carla-simulator/ros-bridge)
- [CARLA_INVS](https://github.com/zijianzhang/CARLA_INVS)

## Citation

> The paper lags behind the code

```
@article{Lai2023ActiveDA,
 title={Active Data Acquisition in Autonomous Driving Simulation},
 author={Jianyu Lai and Zexuan Jia and Boao Li},
 journal={ArXiv},
 year={2023},
 volume={abs/2306.13923}
}
```

# Todo List

- [x] Code and Rules

- [ ] Experiments, and parameters setting

- [ ] Paper work