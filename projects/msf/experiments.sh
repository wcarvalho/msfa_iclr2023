make goto_search group=reward4 search=reward cuda=0,1,2,3
make goto_search group=msf_mask5 search=msf_mask5 cuda=0,1,2,3
make goto_search group=msf_struct5 search=msf_struct5 cuda=0,1,2,3,4


make meta_search searches='usfa' cuda=0,1 # RLDL2

make meta_search searches='usfa_lstm' cuda=0,1,2,3 # RLDL13

make meta_search searches='msf_mask' cuda=0,1,2,3 # RLDL16
make meta_search searches='msf_struct' cuda=0,1,2,3 # RLDL17