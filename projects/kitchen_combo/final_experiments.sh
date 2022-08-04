########################
# Borsa
########################
make final_msf searches='replication' cuda=0,1,2,3 skip=0
# ablations
make final_msf searches='ablate_modularity' cuda=0,1,2,3 skip=0
make final_msf searches='ablate_shared' cuda=0,1,2,3 skip=0


########################
# Fruitbot
########################
make final_fruitbot_taskgen searches='taskgen_final' ray=0 cuda=0,1,2,3 skip=0
make final_fruitbot_procgen searches='procgen_final' ray=0 cuda=0,1,2,3 skip=0


########################
# Minihack
########################
make final_minihack searches='large_final-1' cuda=0,1 skip=0
make final_minihack searches='large_final-2' cuda=0,1 skip=0
make final_minihack searches='small_final' cuda=0,1,2,3 skip=0


########################
# Kitchen Combo
########################
make final_combo searches='test_final' cuda=0,1,2,3 skip=1
make final_combo searches='small_final' cuda=0,1,2,3 skip=1
make final_combo searches='medium_final' cuda=0,1,2,3 skip=0
