make train_search group=uvfa3 search=r2d1

make train_search group=usfa_lstm1 search=usfa_lstm


make train_search group=tasks1 search=r2d1
make train_search group=tasks1 search=usfa_lstm

make train_search group=msf_tasks1 search=msf
make train_search group=msf_modules2 search=msf_modules cuda=0,1,2,3


make train_search group=monolithic2 search=monolithic cuda=0,1,2,3

make train_search group=moredists2 search=moredists cuda=0,1,2,3

make train_search group=lesslang2 search=lesslang cuda=0,1,2,3

make train_search group=bigroom_moredists2 search=bigroom_moredists cuda=0,1,2,3



make train_search group=Clean_Cook_Slice1 search=Clean_Cook_Slice cuda=0,1,2,3,4


make train_search group=multiv4 search=multiv4 cuda=0,1,2,3,4,5,6,7


make train_search group=uvfa4 search=uvfa cuda=0,1,2,3,4,5,6,7

make train_search group=capacity1 search=capacity cuda=0,1,2,3,4,5,6,7