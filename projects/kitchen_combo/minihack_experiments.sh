make train_search_minihack project=minihack search=r2d1 cuda=0,1 ray=1 terminal='current_terminal'
make train_search_minihack project=minihack search=usfa cuda=0,1 ray=1 terminal='current_terminal'
make train_search_minihack project=minihack search=msf cuda=0,1 ray=1 terminal='current_terminal'

make train_search_minihack project=minihack search=r2d1 cuda=0,1,2,3 ray=1 terminal='current_terminal'
make train_search_minihack project=minihack search=usfa cuda=0,1,2,3 ray=1 terminal='current_terminal'
make train_search_minihack project=minihack search=msf cuda=0,1,2,3 ray=1 terminal='current_terminal'


########################
# Final
########################
make final_minihack searches='large_final-1' cuda=0,1
make final_minihack searches='large_final-2' cuda=0,1
make final_minihack searches='small_final' cuda=0,1,2,3
