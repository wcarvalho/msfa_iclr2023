make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=test12_baselines cuda=0,1,2,3 agent=r2d1


make meta_search group=long_fixed searches=long_r2d1 cuda=0,1

make meta_search group=long_fixed searches=long_fixed agent=usfa_lstm cuda=0,1,2,3 terminal='current_terminal'
make meta_search group=long_fixed searches=long_fixed agent=msf cuda=0,1,2,3 terminal='current_terminal'
