agent?=msf
cuda?=2,3
folder?='results/kitchen_grid/final'
wandb?=True
wandb_test?=False
custom_loggers?=True
date?=False
num_gpus?=1
project?='kitchen_grid3'
env?='EasyPickup'
group?=''
notes?=''


export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

test_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/train.py \
	--agent $(agent) \
	--test=True \
	--env_setting=$(env) \
	--wandb=$(wandb_test) \


test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/train_distributed.py \
	--agent $(agent) \
	--wandb=$(wandb_test) \


train_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_gridworld/train_hp_search.py \
	--folder $(folder) \
	--num_gpus $(num_gpus) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes)


train_search_rldl:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_gridworld/train_hp_search.py \
	--spaces rldl_search \
	--folder $(folder) \
	--num_gpus $(num_gpus) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes)




mprof_train_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	mprof run -M projects/kitchen_gridworld/train_hp_search.py \
	--folder $(folder) \
	--group $(group) \
	--wandb=$(wandb) \
	--date=$(date) \
	--num_gpus $(num_gpus) \
	--search $(search) \
	--notes $(notes)
