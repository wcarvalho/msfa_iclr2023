make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=gen5_modr2d1 cuda=0,1,2,3 terminal='current_terminal'



make meta_search agent=msf searches='slice5 cook5 similar5 multiv9_5' cuda=0,1,2,3 skip=0
make meta_search agent=r2d1 searches='slice5 cook5' cuda=0,1,2,3
make meta_search agent=r2d1 searches='similar5 multiv9_5' cuda=0,1,2,3
