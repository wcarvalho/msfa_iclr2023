make goto_search group=simple_model search=usfa_farm_model cuda=2,3 \
  notes="'seeing if independent farm works well without delta-model...'"

make goto_search group=farmflat search=usfa_farmflat_model cuda=2,3 \
  notes="'now seeing if farmflat also does better with just time_contrast...'"

make goto_search group=simple_model search=usfa_farm_model cuda=2,3 \
  notes="'seeing higher reward coeff helps... + no extra time negatives'"

make goto_search group=relational_v1 search=usfa_farm cuda=0,2,3 \
  notes="'seeing if relational helps'"
make goto_search group=relational_v1 search=msf cuda=2,3 \
  notes="'msf: seeing if relational helps'"

make goto_search group=relational_v3 search=msf cuda=0,1,2,3 \
  notes="'msf: delta_concat + residual'"

make goto_search group=relational_v5 search=msf cuda=0,1 \
  notes="'msf: GRU gate + (h or delta)-model '"


make goto_search group=relational_v6 search=msf cuda=0,1,2,3 \
  notes="'does adding mlp help?'"
make goto_search group=relational_v6 search=msf cuda=2,3 \
  notes="'does adding mlp help? how about more heads?'"


make goto_search group=relational_v7 search=msf cuda=0,1,2,3 \
  notes="'does adding mlp help?'"
make goto_search group=relational_v7 search=msf cuda=2,3 \
  notes="'does adding mlp help? how about more heads?'"



make goto_search project=msf_v3 group=baselines_v2 search=usfa cuda=2,3




#--------------------------------------------------------


make goto_search project=msf2 group=q_ablate3 search=q_ablate cuda=0,1
make goto_search project=msf2 group=baselines_v2 search=r2d1
make goto_search project=msf2 group=gate_ablate1 search=gate_ablate cuda=0,1,2,3
make goto_search project=msf2 group=usfa_lstm1 search=usfa_lstm
make goto_search project=msf2 group=relate_ablate2 search=relate_ablate cuda=0,1,2,3
make goto_search project=msf2 group=model_ablate2 search=model_ablate cuda=0,1,2,3
make goto_search project=msf2 group=delta_ablate1 search=delta_ablate cuda=0,1,2,3
make goto_search project=msf2 group=r2d1_no_task1 search=r2d1_no_task cuda=0,1,2,3
make goto_search project=msf2 group=model_ablate4 search=model_ablate cuda=0,1,2,3

make goto_search project=msf2 group=relate_ablate4 search=relate_ablate cuda=0,1,2,3


make goto_search project=msf2 group=baselines3 search=baselines cuda=0,1,2,3