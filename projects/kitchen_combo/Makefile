agent?=msf


cuda?=2,3
wandb?=True
custom_loggers?=True
date?=False
num_gpus?=1

search?='baselines'
terminal?='output_to_files'
group?=''
notes?=''
skip?=1
spaces?='spaces'
ray?=0
debug?=0



export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/:${HOME}/miniconda3/lib

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_DISABLE_JIT=$(nojit)

killall:
	kill -9 $(pgrep ray); kill -9 $(pgrep process_entry); kill -9 $(pgrep python); kill -9 $(pgrep wandb); kill -9 $(pgrep train)


nojit?=0
eval?=1
wandb_test?=False
env?='fruitbot'
setting?=''
init?=0

test_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_combo/train.py \
	--agent=$(agent) \
	--env=$(env) \
	--env_setting=$(setting) \
	--evaluate=$(eval) \
	--test=True \
	--wandb=$(wandb_test)

test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_combo/train_distributed.py \
	--agent $(agent) \
	--env $(env)


project?='kitchen_combo'
train_search_combo:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_combo/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--spaces combo_spaces \
	--env kitchen_combo \
	--terminal $(terminal) \
	--agent $(agent) \
	--debug_search $(debug)

project?='fruitbot'
train_search_fruitbot:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_combo/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--spaces fruitbot_spaces \
	--env fruibot \
	--agent $(agent) \
	--debug_search $(debug)

project?='minihack'
train_search_minihack:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_combo/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--spaces minihack_spaces \
	--env minihack \
	--agent $(agent) \
	--debug_search $(debug)


project?='kitchen_combo'
final_combo:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_combo/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--spaces combo_spaces \
	--env kitchen_combo \
	--terminal $(terminal) \
	--searches "${searches}" \
	--agent $(agent) \
	--debug_search $(debug)


project?='fruitbot'
final_fruitbot:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/kitchen_combo/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces fruitbot_spaces \
	--env fruibot \
	--agent $(agent) \
	--debug_search $(debug)

project?='minihack'
final_minihack:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces minihack_spaces \
	--env minihack \
	--agent $(agent) \
	--debug_search $(debug)

meta_search:
	cd ../..; \
	echo ${searches}; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--wandb=$(wandb) \
	--python_file="projects/kitchen_combo/train_search.py" \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--spaces $(spaces) \
	--searches "${searches}" \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--debug_search $(debug) \
	--agent $(agent)


overwrite?=1
ckpts?=-1
eps?=5
generate_fruitbot_data:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/kitchen_combo/create_analysis_data.py \
	--overwrite=$(overwrite) \
	--ckpts=$(ckpts) \
	--num_episodes=$(eps) \
	--video_path='./results/fruitbot/videos'
