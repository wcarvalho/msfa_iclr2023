# Creating new objects (entire directory)
- new `agents` that use new builder
- new `builder` that uses new learner + actor
- new `learner`
    - 1 class for USFA
    - 1 class for my family of methods
- new `actor`? potentially to deal with randomness
- new `config`


# Dealing with randomness

1. projects/goto: `agent = r2d2.R2D2`
1. r2d2/local_layout: `builder.`make_learner` (accepts key)
1. r2d2/builder: `r2d2_learning.R2D2Learner` (accepts key)
1. r2d2/learning: `unroll.init(key_init, initial_state)
<!-- 1. r2d2/networks: `unroll_hk.init(rng...)` -->
1. usfa/utils: `make_usfa_networks`



# Dealing with acting

1. projects/goto: `loop.run(FLAGS.num_episodes)`
1. r2d2/agent: `self._actor.select_action`
1. agents/jax/actors: `self._policy(self._params, observation, self._state)`
1. r2d2/actor: `recurrent_policy(params, policy_rng, observation, recurrent_state, state.epsilon)`
1. r2d2/networks: ` q_values, core_state = networks.forward.apply`

1. usfa/utils: `make_usfa_networks`

