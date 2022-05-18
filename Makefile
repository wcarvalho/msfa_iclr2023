check?=0
check_end?=0
missions?=25
roomsize?=7
dists?=5
tasks?="cook"

export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

sample_kitchen:
	python envs/babyai_kitchen/sample_kitchen_episodes.py \
	--check $(check) \
	--check-end $(check_end) \
	--missions $(missions) \
	--room-size $(roomsize) \
	--num-distractors $(dists) \
	--task-kinds $(tasks) \
