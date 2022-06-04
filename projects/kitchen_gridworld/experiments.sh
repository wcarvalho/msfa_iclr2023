
make train_search group=similar4 search=similar4 cuda=0,1
make train_search group=genv4 search=genv4 cuda=0,1

make lp_search search=similar4 cuda=0,1

make lp_search search=capacity5 cuda=0,1,2,3
make lp_search search=similar4 cuda=0,1,2,3
make lp_search search=conv_msf cuda=0,1,2,3