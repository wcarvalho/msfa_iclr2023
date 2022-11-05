# Modular Successor Features

- train x not distributed : `goto.py`
- train x distributed : `goto_distributed.py`
- hyperparameter search x distributed: `goto_search.py`


# Adding new baselines

Change:
1. `configs.py`: add config for agent if needed

2. `helpers:load_agent_settings()`: 
    add new elif clause

3. `nets`: add function to create network
