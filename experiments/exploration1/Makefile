agent?=msf
cuda?=2,3
folder?='results/msf2'
wandb?=True
wandb_test?=False
custom_loggers?=True
date?=False
num_gpus?=1
project?='msf2'
search?='baselines'
group?=''
notes?=''
skip?=1
eval?=0
spaces?='borsa_spaces'
ray?=1
cpus?=4
debug?=0
terminal?='current_terminal'
nojit?=0

export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/msfa/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_DISABLE_JIT=$(nojit)

killall:
	kill -9 $(pgrep ray); kill -9 $(pgrep process_entry); kill -9 $(pgrep python); kill -9 $(pgrep wandb); kill -9 $(pgrep goto)
test_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/goto.py \
	--agent $(agent) \
	--test=True \
	--evaluate=$(eval) \
	--wandb=$(wandb_test)

test_async:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/goto_distributed.py \
	--agent $(agent) \
	--debug=True \
	--wandb=$(wandb_test)

goto_search:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/msf/goto_search_lp.py \
	--folder $(folder) \
	--wandb=$(wandb) \
	--date=$(date) \
	--wandb_project $(project) \
	--spaces $(spaces) \
	--group $(group) \
	--search $(search) \
	--ray $(ray) \
	--num_cpus $(cpus) \
	--skip $(skip) \
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
	--spaces $(spaces) \
	--searches "${searches}" \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--num_cpus $(cpus) \
	--debug_search $(debug) \
	--agent $(agent)


final_msf:
	cd ../..; \
	echo "${searches}"; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="projects/msf/goto_search_lp.py" \
	--wandb_project 'msf2' \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--num_cpus $(cpus) \
	--spaces borsa_spaces \
	--folder 'results/msf2' \
	--terminal $(terminal) \
	--searches="${searches}" \
	--agent $(agent) \
	--debug_search $(debug)

evaluate_sync:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/evaluate.py \
	--distributed=True

oracle:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/msf/oracle_performance.py


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
