agent?=msf


cuda?=2,3

date?=False

terminal?='current_terminal'
group?=''
notes?=''
skip?=1
spaces?='spaces'
ray?=0
debug?=0

wandb?=False
wandb_project?=''
wandb_entity?=''


jupyter_lab:
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/acmejax/lib/ \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	DISPLAY=$(cuda) \
	jupyter lab --port $(port) --no-browser --ip 0.0.0.0


final_babyai:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="projects/kitchen_combo/train_search.py" \
	--wandb_project ${wandb_project} \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces borsa_spaces \
	--env goto \
	--folder 'results/${wandb_project}' \
	--agent $(agent) \
	--debug_search $(debug)


final_procgen:
	cd ../..; \
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="projects/kitchen_combo/train_search.py" \
	--wandb_project ${wandb_project} \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces fruitbot_spaces \
	--env fruitbot \
	--folder 'results/${wandb_project}' \
	--agent $(agent) \
	--debug_search $(debug)


final_minihack:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python projects/common/train_search_meta.py \
	--wandb=$(wandb) \
	--date=$(date) \
	--python_file="projects/kitchen_combo/train_search.py" \
	--wandb_project ${wandb_project} \
	--group $(group) \
	--notes $(notes) \
	--skip $(skip) \
	--ray $(ray) \
	--terminal $(terminal) \
	--searches "${searches}" \
	--spaces minihack_spaces \
	--env minihack \
	--folder 'results/${wandb_project}' \
	--agent $(agent) \
	--debug_search $(debug)
