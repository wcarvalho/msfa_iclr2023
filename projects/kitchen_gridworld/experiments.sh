make lp_search search=pickup4 agent=msf cuda=0,1,2,3


make lp_search search=pickup_lang3 cuda=0,1,2,3


make lp_search search=pickup6 agent=r2d1 ray=0 terminal='current_terminal' cuda=0,1,2,3


make meta_search agent=msf searches='slice5 cook5 similar5 multiv9_5' cuda=0,1,2,3 skip=0
make meta_search agent=r2d1 searches='slice5 cook5' cuda=0,1,2,3
make meta_search agent=r2d1 searches='similar5 multiv9_5' cuda=0,1,2,3
