make train_search_rldl group=uvfa3 search=r2d1

make train_search_rldl group=usfa_lstm1 search=usfa_lstm


make train_search_rldl group=tasks1 search=r2d1
make train_search_rldl group=tasks1 search=usfa_lstm

make train_search_rldl group=msf_tasks1 search=msf
make train_search_rldl group=msf_modules2 search=msf_modules cuda=0,1,2,3


make train_search_rldl group=baselines2 search=baselines cuda=0,1,2,3