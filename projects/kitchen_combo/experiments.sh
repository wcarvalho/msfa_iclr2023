make train_search search=baselines cuda=0,1,2,3
make train_search search=msf_reward cuda=0,1,2,3
make train_search search=msf_farm cuda=0,1,2,3


make meta_search searches='usfa' cuda=0,1 # RLDL2

make meta_search searches='usfa_lstm' cuda=0,1,2,3 # RLDL13

make meta_search searches='msf_mask' cuda=0,1,2,3 # RLDL16
make meta_search searches='msf_struct' cuda=0,1,2,3 # RLDL17

