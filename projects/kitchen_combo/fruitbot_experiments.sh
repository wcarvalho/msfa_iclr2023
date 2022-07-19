# Procgen
make train_search_fruitbot search=r2d1_procgen_easy cuda=0,1 ray=1


# Taskgen
make train_search_fruitbot project=fruitbot-taskgen search=r2d1_taskgen_easy cuda=0,1,2,3 ray=1 terminal='current_terminal'
make train_search_fruitbot project=fruitbot-taskgen search=usfa_taskgen_easy cuda=0,1,2,3 ray=1 terminal='current_terminal'
make train_search_fruitbot project=fruitbot-taskgen search=msf_taskgen_easy cuda=0,1,2,3 ray=1 terminal='current_terminal'

# make train_search search=msf_reward2 cuda=0,1,2,3 terminal=current_terminal, ray=1


# make meta_search searches='r2d1_norest' cuda=0,1 # RLDL2

# make meta_search searches='test12_relate' cuda=0,1,2,3

# make meta_search searches='test_noreset' agent='msf' cuda=2,3
