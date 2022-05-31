make train_search group=uvfa3 search=r2d1

make train_search group=clean1 search=clean cuda=0,1,2,3

make train_search group=multi6 search=multi cuda=0,1,2,3
make train_search group=reward5 search=reward_msf cuda=0,1,2,3
make train_search group=phi3 search=phi cuda=0,1,2,3
make train_search group=gen4 search=gen cuda=0,1,2,4 cpus=5
make train_search group=reward_usfa search=reward_usfa cuda=0,1,2,3
make train_search group=mods2 search=mods2 cuda=0,1,2,4


make train_search group=cov5 search=cov5 cuda=0,1,2,4
make train_search group=genv3_3 search=genv3_2 cuda=0,1,2,3
make train_search group=phi8 search=phi8 cuda=0,1,2,4
make train_search group=slice search=slice cuda=0,1

make train_search group=size3 search=size3 cuda=0,1,2,4