make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=test7 cuda=0,1,2,3,4,5,6,7



make meta_search agent=msf searches='gen6_r2d1' agent='r2d1' cuda=0,1
make meta_search agent=msf searches='gen6_r2d1' agent='modr2d1' cuda=0,1


make meta_search agent=msf searches='gen6_usfa_verbose' cuda=0,1,2,3
make meta_search agent=msf searches='gen6_msf_reward' cuda=0,1,2,3
make meta_search agent=msf searches='gen6_msf_size_r10' cuda=0,1,2,3
make meta_search agent=msf searches='gen6_msf_size_r50' cuda=0,1,2,3

