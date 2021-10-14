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
jax_cuda=jax[cuda102]

if [[ $arch = 'gpu' ]]; then
  pip install --upgrade $jax_cuda -f https://storage.googleapis.com/jax-releases/jax_releases.html
fi

# TEST
python projects/starter/dqn_bsuite.py
