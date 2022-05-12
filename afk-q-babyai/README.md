# AFK for Q-BabyAI

# Overview
This is the code for AFK on Q-BabyAI.

### Platform and dependencies:
* Platform: Ubuntu 16.04
* GPU: GEFORCE GTX 1080
* Conda 4.8.3
* Dependencies:
    - Python 3.6
    - Pytorch 1.8.0
    - blosc 1.10.4
    - [OpenAI gym 0.18.3](https://github.com/openai/gym)
    - [gym-minigrid](https://github.com/maximecb/gym-minigrid.git)

### Installation
Install BabyAI
```
cd afk
pip install --editable .
```


# Usage
To reproduce the AFK results on all Q-BabyAI tasks, the ablation study, and the generalization study, please run the following:
```shell
cd run
sh afk_qbabyai.sh
```

For running the baselines, please see the configuration files in ```run/config```
