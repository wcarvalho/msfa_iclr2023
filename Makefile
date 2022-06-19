step?=0
check?=25
missions?=50
roomsize?=7
dists?=0
train?=1
verb?=2
tasks?="cook"
mtasks?="genv3"

cuda?=3

export PYTHONPATH:=$(PYTHONPATH):.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true


sample_kitchen:
	python envs/babyai_kitchen/sample_kitchen_episodes.py \
	--check $(step) \
	--check-end $(check) \
	--missions $(missions) \
	--room-size $(roomsize) \
	--verbosity $(verb) \
	--num-distractors $(dists) \
	--task-kinds $(tasks) \

sample_mkitchen:
	python envs/babyai_kitchen/sample_multilevel_kitchen_episodes.py \
	--check $(step) \
	--check-end $(check) \
	--missions $(missions) \
	--room-size $(roomsize) \
	--verbosity $(verb) \
	--train $(train) \
	--tasks "envs/babyai_kitchen/tasks/v1/$(mtasks).yaml" \

sample_kitchen_pdb:
	python -m ipdb -c continue envs/babyai_kitchen/sample_kitchen_episodes.py \
	--check $(step) \
	--check-end $(check) \
	--missions $(missions) \
	--room-size $(roomsize) \
	--verbosity $(verb) \
	--num-distractors $(dists) \
	--task-kinds $(tasks) \

sample_mkitchen_pdb:
	python -m ipdb -c continue envs/babyai_kitchen/sample_multilevel_kitchen_episodes.py \
	--check $(step) \
	--check-end $(check) \
	--missions $(missions) \
	--room-size $(roomsize) \
	--verbosity $(verb) \
	--train $(train) \
	--tasks "envs/babyai_kitchen/tasks/v1/$(mtasks).yaml" \

jupyter_lab:
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/ \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	DISPLAY=$(cuda) \
	jupyter lab --port 9999 --no-browser --ip 0.0.0.0
