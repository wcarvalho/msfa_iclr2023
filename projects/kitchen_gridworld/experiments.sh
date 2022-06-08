make lp_search search=pickup2 agent=r2d1 cuda=1
make lp_search search=pickup_lang agent=r2d1 cuda=1

make lp_search search=similar4_msf cuda=1

# make lp_search search=multiv9 cuda=0,1 ray=1
make lp_search search=multiv9 cuda=1


make lp_search search=pickup_lang cuda=0,1,2,3
make lp_search search=place_sliced cuda=0,1,2,3 ray=1