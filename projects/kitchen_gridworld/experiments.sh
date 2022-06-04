
make train_search group=cook4 search=cook4 cuda=0,1
make train_search group=genv4 search=genv4 cuda=0,1
make train_search group=cook4 search=cook4 cuda=0,1,2,4
make train_search group=vocab_fix search=vocab_fix cuda=0,1,2,4


make lp_search search=relations cuda=0,1,2,3,4,5