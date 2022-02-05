# rljax
ACME based experiments


# Installation 

```bash
git clone https://github.com/wcarvalho/rljax.git
conda env create --force -f gpu.yaml
```

# Personal

```bash
# GPU Usage
gpustat -i

# killing processed by tune
kill -9 $(pgrep process_entry)
```



## Servers

For multiple GPUS, use `XLA_PYTHON_CLIENT_PREALLOCATE=false` or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.24`



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

