# rljax
ACME based experiments


# Installation 

```bash
git clone https://github.com/wcarvalho/rljax.git --recursive
conda env create --force -f gpu.yaml
```

# Personal

## Brain
```bash

export PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/

sshfs deeplearn9:/home/wcarvalh/projects/rljax/ rlax-brain
```


## RLDL
```bash
sshfs rldl4:/shared/home/wcarvalh/projects/rljax/ rlax-rldl
```

