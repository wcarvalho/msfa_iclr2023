if [[ $1 = gpu ]]; then
  arch=gpu
elif [[ $1 = cpu ]]; then
  arch=cpu
else
  echo 'incorrect arg'
  exit
fi

conda env create --force -f $arch.yaml
eval "$(conda shell.bash hook)"
conda activate acmejax

# CHANGE
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install dm-acme
pip install dm-acme[envs]

# TEST
python projects/starter/dqn_bsuite.py
