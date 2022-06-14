agent?=msf
cuda?=2,3
folder?='results/msf/final'
wandb?=True
custom_loggers?=True
date?=False
num_gpus?=1
project?='msf2'
search?='baselines'
group?=''
notes?=''
skip?=1
ray?=0


export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

test_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/goto.py \
	--agent $(agent) \
	--test=True

test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/goto_distributed.py \
	--agent $(agent)

goto_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/msf/goto_search_lp.py \
	--folder $(folder) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--search $(search) \
	--notes $(notes)

meta_search:
	cd ../..; \
	echo ${searches}; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/msf/goto_search_meta.py \
	--folder $(folder) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--group $(group) \
	--searches "${searches}" \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--agent $(agent)

evaluate_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/evaluate.py \
	--distributed=True


mprof_goto_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	mprof run -M projects/msf/goto_search.py \
	--folder $(folder) \
	--group $(group) \
	--wandb=$(wandb) \
	--date=$(date) \
	--num_gpus $(num_gpus) \
	--search $(search) \
	--notes $(notes)


mprof_goto_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	mprof run -M projects/msf/goto_distributed.py \
	--agent $(agent) \
	--custom_loggers=$(custom_loggers) \
	--debug=True

mprof_kitchen:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	mprof run -M projects/leaks/test_kitch_distributed.py \
	--agent $(agent) \
