make lp_search search=gen5 cuda=0,1 terminal='current_terminal'


make lp_search search=test7 cuda=0,1,2,3,4,5,6,7




make meta_search agent=msf searches='test8' cuda=0,1,2,3,4,5,6,7 terminal='current_terminal' agent=genv5

make meta_search agent=msf searches='test9_gen5_gru' cuda=0,1,2,3 terminal='current_terminal'
make meta_search agent=msf searches='test9_gen5_sum' cuda=0,1,2,3 terminal='current_terminal'
make meta_search agent=msf searches='test9_simple_gru' cuda=0,1,2,3 terminal='current_terminal'