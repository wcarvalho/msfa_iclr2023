# rljax
ACME based experiments


# Installation 

```bash
git clone https://github.com/wcarvalho/rljax.git --recursive
conda env create --force -f gpu.yaml
```

# Personal

```bash
# GPU Usage
gpustat -i

# Updating
git submodule foreach git pull origin main
git pull

# Adding modules
git submodule add https://github.com/deepmind/acme libs/acme
```



## Servers

### Brain
```bash

export PYTHONPATH=$PYTHONPATH:$HOME/projects/rljax/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/acmejax/lib/

sshfs deeplearn9:/home/wcarvalh/projects/rljax/ rlax-brain

tmux kill-session -t launchpad


--lp_launch_type=local_mp

```


### RLDL
```bash
sshfs rldl4:/shared/home/wcarvalh/projects/rljax/ rlax-rldl
```

