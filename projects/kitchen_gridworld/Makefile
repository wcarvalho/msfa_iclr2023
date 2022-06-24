# Training
agent?=msf
eval?=1
custom_loggers?=True
task_reps?='object_verbose'
env?='genv3'
init?=0

# Envivornmnet
cuda?=3
nan?=0
nojit?=0

# Wandb
folder?='results/kitchen_grid/final'
wandb?=True
wandb_test?=False
date?=False
project?='kitchen_grid3'
group?=''
name?=''

# Searching
gpus?=1
cpus?=5
actors?=3
skip?=0
ray?=0
debug?=0
terminal?='output_to_files'
spaces?='brain_search'




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


lp_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_gridworld/train_search.py \
	--folder $(folder) \
	--num_gpus $(gpus) \
	--num_cpus $(cpus) \
	--actors $(actors) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--agent $(agent) \



meta_search:
	cd ../..; \
	echo ${searches}; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--folder $(folder) \
	--wandb=$(wandb) \
	--python_file="projects/kitchen_gridworld/train_search.py" \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--spaces $(spaces) \
	--searches "${searches}" \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--debug_search $(debug) \
	--agent $(agent)


or?=1
ckpts?=-1
eps?=5
generate_analysis_data:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_gridworld/create_analysis_data.py \
	--overwrite=$(or) \
	--ckpts=$(ckpts) \
	--num_episodes=$(eps)
