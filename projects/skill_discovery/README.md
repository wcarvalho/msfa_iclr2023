
# Running experiment

From this directory, call:
```
make train
```

Steps:
1. Run R2D1 on Pickup X
2. Get logging working
3. Running Modular Successor Features on Pickup X
4. Setup statistics for no reward setting:
    a. counts of object-states
    b. # of times objects are picked up
5. Run random agent (lower bound)
6. Run R2D1 again with statistics (tells you minimum time to learn to pickup with reward)
7. Setup VISR = Universal Successor Features + learning dot(phi(state), goal))
    a. basically adding loss
    b. run and see performance
8. Setup VISR version of Modular Successor Features
    a. run and see performance
  
PARALLEL:
    a. double check if newer papers on skill discovery with successor features
