make train_search search=test12_relate cuda=0,1 ray=1
make train_search search=msf_reward2 cuda=0,1,2,3 terminal=current_terminal, ray=1


make meta_search searches='r2d1_norest' cuda=0,1 # RLDL2

make meta_search searches='test12_relate' cuda=0,1,2,3

make meta_search searches='test_noreset' agent='msf' cuda=2,3
