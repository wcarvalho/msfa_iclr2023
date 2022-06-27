make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=test12_baselines cuda=0,1,2,3 agent=r2d1




make meta_search agent=msf searches='test8' cuda=0,1,2,3,4,5,6,7 terminal='current_terminal' agent=genv5

make meta_search agent=msf searches='test10_toggle' cuda=0,1,2,4