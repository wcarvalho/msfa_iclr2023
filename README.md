# Installation 

First, install `miniconda`

```bash
bash setup.sh gpu # gpu based machine
```
Note: ACME only supports GPU based machines :(



# Running experiments

For go to the directory for ICLR, 2023 experiments:
```
cd experiments/iclr2023
conda activate msfa
```
Below are commands for different sets of experiments. 

Notes:

* All results will be under `${root}/results/`
* All results can be viewed with `tensorboard` in their corresponding directory

### BabyAI
Results directory: `${root}/results/borsa`

**main results**

```bash
# experiments defined in `experiments/iclr2023/borsa_spaces.py`
make final_goto searches='main1' cuda=0,1,2,3 #[uvfa, usfa, usfa-learnerd, msfa]
make final_goto searches='main2' cuda=0 #[uvfa-farm]
```

**ablations**
```bash
make final_goto searches='ablate_modularity' cuda=0,1,2,3
make final_goto searches='ablate_gpi' cuda=0,1,2,3
```

### Procgen
Results directory: `${root}/results/fruitbot`

```bash
# experiments defined in `experiments/iclr2023/fruibot_spaces.py`
make final_fruitbot_taskgen searches='main' cuda=0,1,2,3 #[uvfa, uvfa-farm, usfa-learnerd, msfa]
```


### Minihack
Results directory: `${root}/results/minihack`

```bash
# experiments defined in `experiments/iclr2023/minihack_spaces.py`
make final_minihack searches='small' cuda=0,1,2,3 #[uvfa, uvfa-farm, usfa-learnerd, msfa]
make final_minihack searches='large' cuda=0,1,2,3 #[uvfa, uvfa-farm, usfa-learnerd, msfa]
```



