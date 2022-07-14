make train_search_fruitbot search=r2d1_bs cuda=0,1 ray=1
make train_search_fruitbot search=r2d1 cuda=0,1,2,3 ray=1 #R11
make train_search_fruitbot search=r2d1_lr cuda=0,1,2,3 ray=1 #R13
make train_search_fruitbot search=r2d1_nstep cuda=0,1,2,3 ray=1 #18
# make train_search search=msf_reward2 cuda=0,1,2,3 terminal=current_terminal, ray=1


# make meta_search searches='r2d1_norest' cuda=0,1 # RLDL2

# make meta_search searches='test12_relate' cuda=0,1,2,3

# make meta_search searches='test_noreset' agent='msf' cuda=2,3
