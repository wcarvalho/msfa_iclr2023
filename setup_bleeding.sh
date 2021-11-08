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



git clone https://github.com/deepmind/acme.git _acme
cd _acme
git checkout bfab931c67569ee2f796eccb5d094e266767e981 -b compatible
pip install --editable .
cd ..

if [[ $arch = 'gpu' ]]; then
  # TODO: remove 0.2.20 once other libs (tensorflow probability) stop relying on jax.partial
  pip install --upgrade jax[cuda102]==0.2.20 -f https://storage.googleapis.com/jax-releases/jax_releases.html
fi

# TEST
python projects/starter/dqn_bsuite.py
