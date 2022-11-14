make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=gen_long cuda=2,3 ray=1
make lp_search search=gen_long cuda=0,1,2,3 ray=1 agent="msf_4mod_small"


make meta_search group=long_fixed searches=long_baselines cuda=0,1

make meta_search group=long_modr2d1 searches=long_modr2d1 cuda=0,1,2,3 ray=1 