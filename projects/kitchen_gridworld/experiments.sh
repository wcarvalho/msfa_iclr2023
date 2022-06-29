make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=test12_baselines cuda=0,1,2,3 agent=r2d1


make meta_search group=long_fixed searches=long_baselines cuda=0,1,2,3

make meta_search group=long_fixed searches=long_baselines cuda=0,1,2,3
make meta_search group=long_fixed searches=long_msf2 cuda=0,1,2,3

<<<<<<< HEAD
make meta_search agent=msf searches='test9_targets' cuda=0,1,2,4 
=======
make meta_search agent=msf searches='test8' cuda=0,1,2,3,4,5,6,7 terminal='current_terminal' agent=genv5
>>>>>>> c22f4c14503048d361c97dae617a304f0b884682
