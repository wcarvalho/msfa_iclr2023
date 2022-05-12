cuda?=3

export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

jupyter_lab:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	DISPLAY=$(cuda) \
	jupyter lab --port 9999 --no-browser --ip 0.0.0.0

