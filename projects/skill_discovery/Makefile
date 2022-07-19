agent?=r2d1
cuda?=3

export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

train:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	JAX_DEBUG_NANS=True \
	python -m ipdb -c continue projects/skill_discovery/train.py \
	--agent $(agent)

distributed:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue projects/skill_discovery/train_distributed.py \
	--agent $(agent)