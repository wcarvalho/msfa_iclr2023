make train_search search=baselines2 cuda=0,1,2,3
make train_search search=msf_reward2 cuda=0,1,2,3 terminal=current_terminal, ray=1


make meta_search searches='r2d1' cuda=0,1 # RLDL2
make meta_search searches='usfa_lstm' cuda=0,1,2,3 # RLDL16
make meta_search searches='msf_no_mask' cuda=0,1,2,3 terminal=current_terminal # RLDL17

