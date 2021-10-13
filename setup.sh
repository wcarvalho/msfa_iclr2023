conda env create --force -f gpu.yaml

eval "$(conda shell.bash hook)"

conda activate acmejax

# CHANGE
pip install --upgrade "jax[cuda102]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

python projects/starter/dqn_bsuite.py
