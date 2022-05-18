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
task_reps?='pickup'
group?=''
notes?=''
distributed?='False'



export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

git_minor_pull:
	git add -u; git commit -m "updates"; git pull

git_minor_push:
	git add -u; git commit -m "minor"; git push

test_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/train.py \
	--agent $(agent) \
	--test=True \
	--evaluate=True \
	--env_setting=$(env) \
	--task_reps=$(task_reps) \
	--wandb=$(wandb_test) \
	--group=$(group)


test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/train_distributed.py \
	--agent $(agent) \
	--wandb=$(wandb_test) \
	--env_setting=$(env) \
	--task_reps=$(task_reps) \
	--group=$(group)


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



video_maker:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/create_episode_data.py \
	--distributed=$(distributed)

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
