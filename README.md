# Installation 

First, install `miniconda`

```bash
bash setup.sh gpu # gpu based machine
# EXPECTED ERRORS for jax>=0.2.26
# 1. rlax 0.1.1 requires <=0.2.21
# 2. distrax 0.1.0 requires jax<=0.2.21
```
Note: ACME only supports GPU based machines :(



# Running experiments

For go to the directory for ICLR, 2023 experiments:
```
cd experiments/iclr2023  # go to experiments directory
```
Below are commands for different sets of experiments. 

Notes:

* All experiments run a *distributed* setting with actors/evaluators/learner all running fully in parallel (thanks to ACME!).
* All results will be under `${root}/results/`
* All results can be viewed with `tensorboard` in their corresponding directory

#### Test install

To test the install, we can run a non-distributed agent on the baby experiments.

```
make test_sync env=goto agent=msf
```






### BabyAI

Results directory: `${root}/results/borsa`

**main results**

```bash
# experiments defined in `experiments/iclr2023/borsa_spaces.py`
conda activate msfa  # activate environment
make final_goto searches='main1' cuda=0,1,2,3 #[uvfa, usfa, usfa-learnerd, msfa]
make final_goto searches='main2' cuda=0 #[uvfa-farm]
```

**ablations**

```bash
conda activate msfa  # activate environment
make final_goto searches='ablate_modularity' cuda=0,1,2
make final_goto searches='ablate_gpi' cuda=0,1,2
```



### Procgen

Results directory: `${root}/results/fruitbot`

```bash
# experiments defined in `experiments/iclr2023/fruibot_spaces.py`
conda activate msfa  # activate environment
make final_fruitbot_taskgen searches='main' cuda=0,1,2,3 #[uvfa, uvfa-farm, usfa-learnerd, msfa]
```



### Minihack

Results directory: `${root}/results/minihack`

```bash
# experiments defined in `experiments/iclr2023/minihack_spaces.py`
conda activate msfa  # activate environment
make final_minihack searches='small' cuda=0,1,2,3 #[uvfa, uvfa-farm, usfa-learnerd, msfa]
make final_minihack searches='large' cuda=0,1,2,3 #[uvfa, uvfa-farm, usfa-learnerd, msfa]
```



