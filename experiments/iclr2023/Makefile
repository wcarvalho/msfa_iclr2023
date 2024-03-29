agent?=msf


cuda?=2,3
wandb?=False
custom_loggers?=True
date?=False
num_gpus?=1

search?='baselines'
terminal?='current_terminal'
group?=''
notes?=''
skip?=1
spaces?='spaces'
ray?=0
debug?=0



export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/msfa/lib/:${HOME}/miniconda3/lib

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
	python -m ipdb -c continue experiments/iclr2023/train.py \
	--agent=$(agent) \
	--env=$(env) \
	--env_setting=$(setting) \
	--evaluate=$(eval) \
	--test=True \
	--init_only=$(init) \
	--wandb=$(wandb_test)

test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/iclr2023/train_distributed.py \
	--agent $(agent) \
	--env $(env)

train_search_goto:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/iclr2023/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project 'msf2' \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--spaces borsa_spaces \
	--env goto \
	--terminal $(terminal) \
	--agent $(agent) \
	--debug_search $(debug)


train_search_combo:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/iclr2023/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project 'iclr2023' \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--spaces combo_spaces \
	--env iclr2023 \
	--terminal $(terminal) \
	--agent $(agent) \
	--debug_search $(debug)

train_search_fruitbot:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/iclr2023/train_search.py \
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
	--env fruitbot \
	--folder 'results/fruitbot' \
	--agent $(agent) \
	--debug_search $(debug)

train_search_minihack:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/iclr2023/train_search.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project 'minihack' \
	--group $(group) \
	--search $(search) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--spaces minihack_spaces \
	--env minihack \
	--folder 'results/minihack' \
	--agent $(agent) \
	--debug_search $(debug)


final_fruitbot_taskgen:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="experiments/iclr2023/train_search.py" \
	--wandb_project 'fruitbot-taskgen' \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces fruitbot_spaces \
	--env fruitbot \
	--folder 'results/fruitbot' \
	--agent $(agent) \
	--debug_search $(debug)

final_minihack:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="experiments/iclr2023/train_search.py" \
	--wandb_project 'minihack' \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces minihack_spaces \
	--env minihack \
	--folder 'results/minihack' \
	--agent $(agent) \
	--debug_search $(debug)

final_goto:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="experiments/iclr2023/train_search.py" \
	--wandb_project 'borsa' \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces borsa_spaces \
	--env goto \
	--folder 'results/borsa' \
	--agent $(agent) \
	--debug_search $(debug)


meta_search:
	cd ../..; \
	echo ${searches}; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python experiments/common/train_search_meta.py \
	--wandb=$(wandb) \
	--python_file="experiments/iclr2023/train_search.py" \
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
generate_borsa_object_counts:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/iclr2023/borsa_object_counts.py \
	--overwrite=$(overwrite)


overwrite?=1
ckpts?=-1
eps?=5
video_path?='./results/fruitbot/videos'
video_folder?='fruitbot'
generate_fruitbot_data:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue experiments/iclr2023/create_analysis_data.py \
	--overwrite=$(overwrite) \
	--ckpts=$(ckpts) \
	--num_episodes=$(eps) \
	--video_path=$(video_path) \
	--folder=$(video_folder)
