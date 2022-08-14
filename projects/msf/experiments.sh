make goto_search group=reward4 search=reward cuda=0,1,2,3

make goto_search search=msf cuda=0,1,2,3 ray=1
make goto_search search=msf_rate cuda=0,1,2,3 ray=1 cpus=3

# make meta_search searches='usfa' cuda=0,1 # RLDL2
# make meta_search searches='usfa_lstm' cuda=0,1,2,3 # RLDL13
# make meta_search searches='msf_mask' cuda=0,1,2,3 # RLDL16
# make meta_search searches='msf_struct' cuda=0,1,2,3 # RLDL17


########################
# Final
########################
make final_msf searches='replication' cuda=0,1,2,3
make final_msf searches='ablate_modularity' cuda=0,1,2,3
make final_msf searches='ablate_shared_phi_psi' cuda=0,1,2,3
make final_msf searches='ablate_relation_heads' cuda=0,1,2,3
make final_msf searches='ablate_share_attention' cuda=0,1
