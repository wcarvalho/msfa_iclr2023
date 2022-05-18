check?=0
check_end?=25
missions?=50
roomsize?=7
dists?=0
tasks?="cook"

cuda?=3

export PYTHONPATH:=$(PYTHONPATH):.

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

sample_kitchen_pdb:
	python -m ipdb -c continue envs/babyai_kitchen/sample_kitchen_episodes.py \
	--check $(check) \
	--check-end $(check_end) \
	--missions $(missions) \
	--room-size $(roomsize) \
	--num-distractors $(dists) \
	--task-kinds $(tasks) \

jupyter_lab:
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/ \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	DISPLAY=$(cuda) \
	jupyter lab --port 9999 --no-browser --ip 0.0.0.0
