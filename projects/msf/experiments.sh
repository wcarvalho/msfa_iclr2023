make goto_search search=r2d1 cuda=0,1,2,3
make goto_search search=r2d1_farm cuda=0,1,2,3
make goto_search search=usfa cuda=0,1,2,3
make goto_search search=usfa_lstm cuda=0,1,2,3
make goto_search search=msf cuda=0,1,2,3

# make meta_search searches='usfa' cuda=0,1 # RLDL2
# make meta_search searches='usfa_lstm' cuda=0,1,2,3 # RLDL13
# make meta_search searches='msf_mask' cuda=0,1,2,3 # RLDL16
# make meta_search searches='msf_struct' cuda=0,1,2,3 # RLDL17


########################
# Final
########################
make final_msf searches='xl_respawn' ray=0 cuda=0,1,2,3
make final_msf searches='xxl_nopickup' ray=0 cuda=0,1,2,3

make final_msf searches='gate_search' ray=0 cuda=0,1,2,3
make final_msf searches='small_noise_gpi_n' ray=0 cuda=0,1,2,3
make final_msf searches='small_noise_gpi_y' ray=0 cuda=0,1,2,3
make final_msf searches='one_policy_gpi_y' ray=0 cuda=0,1,2,3
