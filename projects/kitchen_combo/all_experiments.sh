################################################
# Borsa
################################################
  make train_search_goto search=r2d1 cuda=0,1,2,3
  make train_search_goto search=r2d1_farm cuda=0,1,2,3
  make train_search_goto search=usfa cuda=0,1,2,3
  make train_search_goto search=usfa_lstm cuda=0,1,2,3
  make train_search_goto search=msf cuda=0,1,2,3


  # ablations
  make final_goto searches='ablate_modularity' cuda=0,1,2,3 skip=0
  make final_goto searches='ablate_shared' cuda=0,1,2,3 skip=0
  make final_goto searches='xl_respawn' cuda=0,1,2,3


  make final_goto searches='random' cuda=0,1,2,3
  make final_goto searches='one_policy_gpi_n' cuda=0,1,2,3
  make final_goto searches='ablate_modularity' cuda=0,1,2,3

################################################
# Fruitbot
################################################
  # Procgen
  make train_search_fruitbot search=r2d1_procgen_easy cuda=0,1,2,3 ray=1

  # Taskgen
  make train_search_fruitbot project=fruitbot-taskgen search=r2d1_taskgen_easy cuda=0,1,2,3 ray=1

  #-----------------------
  # Final
  #-----------------------
  make final_fruitbot_taskgen searches='uvfa_taskgen' cuda=3 ray=0 skip=1
  make final_fruitbot_taskgen searches='sfa_taskgen' cuda=0,1,2,3 ray=0
  make final_fruitbot_taskgen searches='long_sf' cuda=0,1,2,3 ray=0
  make final_fruitbot_taskgen searches='taskgen_final' cuda=0,1,2,3 ray=0


  make final_fruitbot_procgen searches='uvfa_seedgen' cuda=3 ray=0 skip=1
  make final_fruitbot_procgen searches='sfa_seedgen' cuda=0,1,2,3 ray=0
  make final_fruitbot_procgen searches='procgen_final' cuda=0,1,2,3 ray=0


################################################
# Minihack
################################################
  make final_minihack searches='uvfa_small' cuda=0,1,2,3 ray=0
  make final_minihack searches='sfa_small' cuda=0,1,2,3 ray=0
  make final_minihack searches='uvfa_large' cuda=0,1,2,3 ray=0
  make final_minihack searches='sfa_large' cuda=0,1,2,3 ray=0

  make final_minihack searches='random' cuda=0,1,2,3 ray=0

################################################
# Kitchen Combo
################################################
  make train_search_combo search=test_lp cuda=0 ray=1 terminal='current_terminal'

  make final_combo searches='medium_final' cuda=0,1,2,3 skip=0


################################################
# Kitchen Gridworld
################################################
  make final_kitchen searches='uvfa' cuda=0,1,2,3 ray=0
  make final_kitchen searches='sfa' cuda=0,1,2,3 ray=0