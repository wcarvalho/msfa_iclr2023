# rljax
ACME based experiments


# Installation 

```bash
git clone https://github.com/wcarvalho/rljax.git
bash setup.sh gpu_procgen_minihack # gpu based machine
```

# Experiments

### BabyAI
```
# main results
make final_babyai searches='xl_respawn' cuda=0,1,2,3

# ablation: modularity
make final_babyai searches='ablate_modularity' cuda=0,1,2,3

# ablation: GPI
make final_babyai searches='one_policy_gpi_n' cuda=0,1,2,3

```

### Procgen
```
make final_procgen searches='taskgen_final' cuda=0,1,2 ray=0
```

### Minihack
```
make final_minihack searches='small_final' cuda=0,1,2 ray=0
make final_minihack searches='large_final_all' cuda=0,1,2 ray=0
```
