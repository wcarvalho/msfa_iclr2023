make train_search group=uvfa3 search=r2d1

make train_search group=clean1 search=clean cuda=0,1,2,3

make train_search group=multi6 search=multi cuda=0,1,2,3
make train_search group=reward5 search=reward_msf cuda=0,1,2,3
make train_search group=phi3 search=phi cuda=0,1,2,3
make train_search group=gen4 search=gen cuda=0,1,2,4 cpus=5
