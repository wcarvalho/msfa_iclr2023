# Procgen
make train_search_fruitbot search=r2d1_procgen_easy cuda=0,1,2,3 ray=1
make train_search_fruitbot search=usfa_procgen_easy cuda=0,1,2,3 ray=1
make train_search_fruitbot search=msf_procgen_easy cuda=0,1,2,3 ray=1


# Taskgen
make train_search_fruitbot project=fruitbot-taskgen search=r2d1_taskgen_easy cuda=0,1,2,3
make train_search_fruitbot project=fruitbot-taskgen search=usfa_taskgen_easy cuda=0,1,2,3
make train_search_fruitbot project=fruitbot-taskgen search=msf_taskgen_easy1 cuda=0,1,2,3
make train_search_fruitbot project=fruitbot-taskgen search=msf_taskgen_easy2 cuda=0,1,2,3

# make train_search search=msf_reward2 cuda=0,1,2,3 terminal=current_terminal, ray=1

make final_fruitbot_taskgen searches='one_policy' cuda=0,1,2,3

########################
# Final
########################

make final_fruitbot_taskgen searches='taskgen_final' cuda=0,1,2,3
make final_fruitbot_procgen searches='procgen_final' cuda=0,1,2,3
