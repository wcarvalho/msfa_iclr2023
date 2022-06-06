agent?=msf
eval?=1
cuda?=0,1
nan?=0
folder?='results/kitchen_grid/final'
wandb?=True
wandb_test?=False
custom_loggers?=True
date?=False
gpus?=1
cpus?=3
actors?=5
project?='kitchen_grid3'
env?='genv3'
task_reps?='pickup'
group?=''
name?=''
notes?=''
dist?='False'
nojit?=0
init?=0

export WANDB_START_METHOD=thread
export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_DISABLE_JIT=$(nojit)
export JAX_DEBUG_NANS=$(nan)

git_minor_pull:
	git add -u; git commit -m "updates"; git pull

git_minor_push:
	git add -u; git commit -m "minor"; git push

test_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	JAX_DEBUG_NANS=True \
	python -m ipdb -c continue projects/kitchen_gridworld/train.py \
	--agent $(agent) \
	--test=True \
	--evaluate=$(eval) \
	--env_setting=$(env) \
	--task_reps=$(task_reps) \
	--wandb=$(wandb_test) \
	--group=$(group) \
	--init_only=$(init)


test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	JAX_DEBUG_NANS=True \
	python -m ipdb -c continue projects/kitchen_gridworld/train_distributed.py \
	--agent $(agent) \
	--wandb=$(wandb_test) \
	--debug=True \
	--env_setting=$(env) \
	--task_reps=$(task_reps) \
	--group=$(group) \
	--name=$(name)


train_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_gridworld/train_hp_search.py \
	--folder $(folder) \
	--num_gpus $(gpus) \
	--num_cpus $(cpus) \
	--actors $(actors) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes)

lp_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_gridworld/train_hp_search_lp.py \
	--folder $(folder) \
	--num_gpus $(gpus) \
	--num_cpus $(cpus) \
	--actors $(actors) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes)


# train_search_rldl:
# 	cd ../..; \
# 	CUDA_VISIBLE_DEVICES=$(cuda) \
# 	python projects/kitchen_gridworld/train_hp_search.py \
# 	--spaces rldl_search \
# 	--folder $(folder) \
# 	--num_gpus $(gpus) \
# 	--wandb=$(wandb) \
# 	--date=$(date) \
# 	--wandb_project $(project) \
# 	--group $(group) \
# 	--search $(search) \
# 	--notes $(notes)



video_maker:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/create_episode_data.py \
	--distributed=$(dist)

mprof_train_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	mprof run -M projects/kitchen_gridworld/train_hp_search.py \
	--folder $(folder) \
	--group $(group) \
	--wandb=$(wandb) \
	--date=$(date) \
	--num_gpus $(gpus) \
	--search $(search) \
	--notes $(notes)
