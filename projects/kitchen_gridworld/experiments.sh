make train_search group=uvfa3 search=r2d1

make train_search group=clean1 search=clean cuda=0,1,2,3,4,5
make train_search group=multi5 search=multi cuda=0,1,2,3,4
make train_search group=layernorm1 search=layernorm cuda=0,1,2,3,4,5


make train_search group=params1 search=params cuda=0,1,2,3,4,5