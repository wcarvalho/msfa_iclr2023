make train_search search=baselines2 cuda=0,1,2,3
make train_search search=msf_reward2 cuda=0,1,2,3 terminal=current_terminal, ray=1


make meta_search searches='r2d12' cuda=0,1 # RLDL2
make meta_search searches='usfa_lstm2' cuda=0,1,2,3 # RLDL16
make meta_search searches='msf2' cuda=0,1,2,3 terminal=current_terminal # RLDL17

